# GACaps-HTC: Graph Attention Capsule Network for Hierarchical Text Classification

This is the official implementation of [GACaps-HTC: Graph Attention Capsule Network for Hierarchical Text Classification].

### 1. Configuration

`run.py` takes the path to a configuration file as its argument. (e.g. `python run.py configs/cfg.json`)

### 2. Data

- `config.path.data.train`, `config.path.data.val`, and `config.path.data.test`

    - These files are .json files where each line is a JSON object composed as follows:

        `{"text": "this is an example document", "label": ["<study_unit_label_name>", "<parent_topic_label_name>", ancester topics]}`

    - The labels are required to be sorted from the leaf node to the node just below the root.

        e.g. `["Vector, Matrix, and Tensor", "Basics of Linear Algebra", "Linear Algebra", "AI Prerequisites"]`

- `config.path.labels`

    - This is a json file containing a dictionary where label names and label indices are keys and values, respectively.

- `config.path.prior`

    - This is a json file containing a dictionary of the following format:

        `{"<parent_label_name>": {"<child_1_label_name>": <prior 1>, "<child_2_label_name>": <prior 2> ...} ...}`

        e.g. `{"AI Prerequisites": {"Calculus": 0.4, "Linear Algebra": 0.3, "Probability": 0.2, "Statistics": 0.1} ...}`
    
    - For each parent node - child node pair, prior is simply obtained as the proportion of the number of instances.

- `config.path.hierarchy`

    - This is a tsv file where each line is written as follows:

        `<parent_label_name>\t<child_1_label_name>\t<child_2_label_name>...`


### 3. Requirements

`run.py` requires the following libraries: `numpy, torch, transformers`.

