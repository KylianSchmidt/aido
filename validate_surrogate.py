import sys

import pandas as pd

from aido.surrogate import Surrogate, SurrogateDataset
from aido.surrogate_validation import SurrogateValidation
from aido.training import pre_train

if __name__ == "__main__":
    if len(sys.argv) == 2:
        surrogate_dataset = SurrogateDataset(pd.read_parquet(sys.argv[1]), norm_reco_loss=True)

        surrogate = Surrogate(*surrogate_dataset.shape, n_time_steps=200)

        n_epochs_pre = 50
        n_epochs_main = 100
        pre_train(surrogate, surrogate_dataset, n_epochs_pre)
        surrogate.train_model(
            surrogate_dataset,
            batch_size=256,
            n_epochs=n_epochs_main,
            lr=0.0005
        )
        surrogate_loss = surrogate.train_model(
            surrogate_dataset,
            batch_size=256,
            n_epochs=n_epochs_main,
            lr=0.0001
        )
        validator = SurrogateValidation(surrogate)
        validation_df = validator.validate(surrogate_dataset, batch_size=20)
        validation_df.to_parquet(".validation_df")

    else:
        validation_df = pd.read_parquet(".validation_df")
        print("Validation DataFrame found")

    SurrogateValidation.plot(validation_df, ".")
