import torch
from pytest_mock import MockerFixture


def test_training_loop_out_of_memory(mocker: MockerFixture):
    # Mock the training_loop function to simulate OOM errors and success
    mock_training_loop = mocker.patch('aido.training.training_loop')
    mock_training_loop.side_effect = torch.cuda.OutOfMemoryError("OOM error")

    num_training_loop_tries = 0
    training_loop_out_of_memory = True
    reco_paths_dict = {"own_path": "dummy_path"}
    interface = mocker.MagicMock()
    interface.loss = "dummy_loss"
    interface.constraints = "dummy_constraints"

    while training_loop_out_of_memory:
        try:
            training_loop_out_of_memory = False
            mock_training_loop(
                reco_file_paths_dict=reco_paths_dict["own_path"],
                reconstruction_loss_function=interface.loss,
                constraints=interface.constraints
            )
        except torch.cuda.OutOfMemoryError:
            training_loop_out_of_memory = True
            num_training_loop_tries += 1
            torch.cuda.empty_cache()

            if num_training_loop_tries > 10:
                training_loop_out_of_memory = False

    assert num_training_loop_tries == 11
