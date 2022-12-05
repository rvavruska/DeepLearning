
with open('train/labels/detections.csv', 'r') as inp:
    lines = inp.readlines()

# We open the target file in write-mode
with open('train/labels/detections_clean.csv', 'w') as out:
    # We go line by line writing in the target file
    # if the original line does not include the
    # strings 'py-board' or 'coffee'
    for line in lines:
        if '/m/0k4j' in line:
            out.write(line)