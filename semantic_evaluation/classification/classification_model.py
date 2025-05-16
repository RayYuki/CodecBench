import torch
import numpy as np
from typing import List, Union, Tuple
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, average_precision_score
import warnings 
warnings.filterwarnings("ignore")

class Model:
    """
    Model class: this is a base class for all models. It contains the basic methods
    for generating embeddings from audio files.
    """
    def __init__(
        self,
        model,
        **kwargs,
    ):
        self.model = model
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        
    def get_embeddings(
        self,
        audio: np.ndarray,
        **kwargs,
    ):
        """
        Get embeddings from audio. This method should be implemented by the child class.
        It generates embeddings for the whole audio.
        :param audio: audio array
        :return: a tensor of shape (embedding_size,)
        """
        raise NotImplementedError

    def get_token_embeddings(
        self,
        audio: np.ndarray,
        frame_length_ms: int = 20,
        **kwargs,
    ):
        """
        Get token embeddings from audio. This method should be implemented by the child class.
        It generates embeddings for each frame of the audio. The frame length is specified
        by the frame_length_ms parameter.
        :param audio: audio array
        :param frame_length_ms: frame length in milliseconds
        :return: a tensor of shape (n_frames, embedding_size)
        """
        raise NotImplementedError

    def get_classification_embedding_size(self):
        """
        Get the embedding size for classification. This method should be implemented by the child class.
        :return: an integer
        """
        raise NotImplementedError

    def get_token_embedding_size(self):
        """
        Get the embedding size for token embeddings. This method should be implemented by the child class.
        :return: an integer
        """
        raise NotImplementedError

    def get_sampling_rate(self):
        """
        Get the sampling rate. This method should be implemented by the child class.
        :return: an integer
        """
        raise NotImplementedError

class ClassificationModel:
    """
    This class implements a classification model. It contains the basic methods for
    training and evaluating a classification model.
    """
    def __init__(
        self,
        layers: Union[List[int], Tuple[int]],
        input_embedding_size: int,
        activation: str = "relu",
        dropout: float = 0.1,
        num_classes: int = 2,
        verbose: bool = False,
        is_multilabel: bool = False,
        mode: str = "linear",
        codec_model: Model = None,
        **kwargs,
    ):
        """
        :param layers: list of layer sizes
        :param input_embedding_size: size of the input embedding
        :param activation: activation function that will be used for non-linear evaluation
        :param dropout: dropout rate
        :param num_classes: number of classes
        :param verbose: whether to print progress
        """
        self.codec_model = codec_model
        self.layers = layers
        self.input_embedding_size = input_embedding_size
        self.activation = activation
        self.dropout = dropout
        self.num_classes = num_classes
        self.verbose = verbose
        self.is_multilabel = is_multilabel
        self.mode = mode
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.model = self._build_model()

    def _build_model(self):
        """
        Build the model according to the specified parameters.
        :return: a torch.nn.Module
        """

        if len(self.layers) == 0:
            model = [torch.nn.Linear(self.input_embedding_size, self.num_classes)]
        else:
            # Build the model
            model = []
            for i, layer_size in enumerate(self.layers):
                if i == 0:
                    model.append(torch.nn.Linear(self.input_embedding_size, layer_size))
                else:
                    model.append(torch.nn.Linear(self.layers[i - 1], layer_size))
                model.append(torch.nn.Dropout(self.dropout))
                model.append(torch.nn.ReLU())
            model.append(torch.nn.Linear(self.layers[-1], self.num_classes))

        clf_model = torch.nn.Sequential(*model)

        return clf_model

    def train_epoch(
        self,
        train_dataloader,
        optimizer,
        scheduler,
        criterion,
        device,
        **kwargs,
    ):
        """
        Train the model for one epoch.
        :param train_dataloader: training data loader
        :param optimizer: optimizer
        :param criterion: loss function
        :param device: device
        :return: loss
        """
        self.model.train()
        running_loss = 0.0
        for i, (audio, mask, labels) in enumerate(train_dataloader):
            audio = audio.to(device)
            mask = mask.to(device)
            inputs = self.codec_model.get_embeddings(audio, mask)
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = self.model(inputs)

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()
        return running_loss / len(train_dataloader)
    
    def train(
        self,
        train_dataloader,
        val_dataloader,
        learning_rate: float = 1e-3,
        device: str = "cpu",
        max_num_epochs: int = 10,
        **kwargs,
    ):
        """
        Train the model.
        :param train_dataloader: training data loader
        :param val_dataloader: validation data loader
        :param device: device to use for training (cpu or cuda)
        :return: best model and metrics
        """

        # iterate over epochs
        best_model = None
        best_val_loss = np.inf
        best_val_acc = 0.0
        best_val_f1 = 0.0
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, capturable=True)
        global_steps = max_num_epochs * len(train_dataloader)
        warmup_percentage = 0.1
        # make linear warmup schedule and linear decay schedule
        warmup_steps = int(global_steps * warmup_percentage)
        lr_lambda = lambda step: min(
            1.0, step / warmup_steps
        ) if step <= warmup_steps else max(
            0.0, 1.0 - (step - warmup_steps) / (global_steps - warmup_steps)
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.model = self.model.to(device)

        for epoch in tqdm(range(max_num_epochs), desc="Epochs"):
            # train for one epoch
            train_loss = self.train_epoch(
                train_dataloader, optimizer, scheduler, self.criterion, device
            )
            scheduler.step()
            # evaluate on validation set
            metrics = self.evaluate(val_dataloader, device)

            # save best model
            if metrics["loss"] < best_val_loss or best_model is None:
                best_val_metrics = metrics
                best_model = self.model.state_dict()

            # report metrics in tqdm
            if self.verbose:
                str_metrics = " ".join(
                    [f"{k}: {v:.4f}" for k, v in metrics.items()]
                )
                tqdm.write(f"Epoch {epoch + 1} - train loss: {train_loss:.4f} - {str_metrics}")


        # load best model
        self.model.load_state_dict(best_model)
        return best_model, best_val_metrics

    def evaluate(
        self,
        dataloader,
        device,
        **kwargs,
    ):
        """
        Evaluate the model on the given data loader.
        :param data_loader: data loader containing the data to evaluate on
        :param device: device to use for evaluation (cpu or cuda)
        :return: loss, accuracy, f1 score
        """
        self.model.eval()
        running_loss = 0.0
        y_true = []
        y_pred = []
        with torch.no_grad():
            for i, (audio, mask, labels) in enumerate(dataloader):
                audio = audio.to(device)
                mask = mask.to(device)
                labels = labels.to(device)
                inputs = self.codec_model.get_embeddings(audio, mask)
                inputs = inputs.to(device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item()
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(outputs.argmax(dim=1).cpu().numpy())

        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="macro")
        return {
            "loss": running_loss / len(dataloader),
            "accuracy": accuracy,
            "f1": f1,
        }

