import os

# Input and Output paths

script_dir = os.path.dirname(os.path.abspath(__file__))

# INPUT_IMAGES_DIR = os.path.join(
#     script_dir,
#     "..",
#     "input_images"
# )

OUTPUT_TAGS_DIR = os.path.join(
    script_dir,
    "..",
    "output",
    "tagging_output"
)

OUTPUT_DESCRIPTIONS_DIR = os.path.join(
    script_dir,
    "..",
    "output",
    "tagging_output",
    "lvlm_description_output"
)

OUTPUT_LOCATION_DIR = os.path.join(
    script_dir,
    "..",
    "output",
    "location_output"
)

# OUTPUT_SEGMENTATION_DIR = os.path.join(
#     script_dir,
#     "..",
#     "output",
#     "segmentation_output"
# )


# Additional paths

CONFIG_DIR = script_dir

TAGGING_DIRECT_LVLM_PROMPT = os.path.join(
    script_dir,
    "..",
    "talos",
    "tagging",
    "direct_lvlm_tagging",
    "prompts",
    "prompt.txt"
)


TAGGING_LLM_PROMPTS_DIR = os.path.join(
    script_dir,
    "..",
    "talos",
    "tagging",
    "lvlm_llm_tagging",
    "llm_keyword_extraction",
    "prompts"
)

TAGGING_LLM_PROMPT1 = os.path.join(
    TAGGING_LLM_PROMPTS_DIR,
    "prompt1.txt"
)

TAGGING_LLM_PROMPT2 = os.path.join(
    TAGGING_LLM_PROMPTS_DIR,
    "prompt2.txt"
)


# Evaluation paths

EVALUATION_DIR = os.path.join(
    script_dir,
    "..",
    "..",
    "evaluation"
)

# LVIS evaluation input images directory
INPUT_IMAGES_DIR = os.path.join(
    EVALUATION_DIR,
    "datasets",
    "LVIS",
    "images"
)

# LVIS evaluation output segmentation directory
OUTPUT_SEGMENTATION_DIR = os.path.join(
    EVALUATION_DIR,
    "datasets",
    "LVIS",
    "talos",
    "talos_masks"
)
