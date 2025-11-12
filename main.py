# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: python-uppgift01
#     language: python
#     name: python-uppgift01
# ---

# %% [markdown]
# ## CLI helpers for the MNIST demo
# These cells mirror the notebook structure so the script stays readable in
# Jupyter while remaining executable as a plain Python module.

# %%
import matplotlib.pyplot as plt

from mnist import MODEL_STATE_PATH, Model


# %% [markdown]
# ### User input helpers
# Prompt utilities that wrap `input` with defaults and validation.


# %%
def prompt_yes_no(message: str, default: bool = False) -> bool:
    choice = input(message).strip().lower()
    if not choice:
        return default
    return choice in ("y", "yes")


def prompt_sample_index(max_index: int, default: int = 0) -> int:
    while True:
        choice = input(
            f"Enter test sample index (0-{max_index}, default {default}): "
        ).strip()

        if not choice:
            return default

        try:
            value = int(choice)
            return max(0, min(value, max_index))
        except ValueError:
            print("Invalid input, please enter an integer.")


# %% [markdown]
# ### Visualization helper
# Render a grayscale digit with matplotlib when the user asks for it.


# %%
def show_sample_image(image) -> None:
    plt.imshow(image, cmap="gray")
    plt.show()


# %% [markdown]
# ### Training orchestration
# Decide whether to reuse saved weights or retrain from scratch.


# %%
def maybe_retrain_model(epochs: int = 10) -> Model:
    if MODEL_STATE_PATH.exists():
        choice = (
            input(
                f"Existing weights found at {MODEL_STATE_PATH}. "
                "Press Enter to use them, or type 'r' to retrain: "
            )
            .strip()
            .lower()
        )
        if choice == "r":
            model = Model()
            model.train(epochs)
        else:
            model = Model.load(MODEL_STATE_PATH)
    else:
        print("No trained weights found, starting a new training run.")
        model = Model()
        model.train(epochs)

    return model


# %% [markdown]
# ### Interactive inference loop
# Continuously classify samples until the user opts out.


# %%
def inference_loop(model: Model) -> None:
    max_index = model.get_max_test_index()
    default_index = 0

    while True:
        sample_index = prompt_sample_index(max_index, default_index)
        result = model.classify(sample_index)
        print(f"Prediction for sample {result['index']}: {result['prediction']}")

        if prompt_yes_no("Display the digit with matplotlib? [y/N]: "):
            show_sample_image(result["image"])

        if not prompt_yes_no("Classify another test sample? [y/N]: "):
            break


# %% [markdown]
# ### Entry point
# Train/load once, then enter the inference loop.


# %%
def main():
    model = maybe_retrain_model()
    inference_loop(model)


if __name__ == "__main__":
    main()

# %%
