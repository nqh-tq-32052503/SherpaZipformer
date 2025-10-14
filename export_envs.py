import os 

env_list = {
    "TRAIN_CUTS" : "",
    "VALID_CUTS" : "",
    "MATERIAL_DIR" : "./pseudo_data",
    "CHECKPOINT_PATH" : "./pretrained.pt",
    "FREEZE_MODULES" : "encoder",
    "SAVE_DIR" : "./outputs",
    "MAX_DURATION" : "600",
    "DEVICE" : "1"
}

for key in env_list:
    os.environ[key] = str(env_list[key])