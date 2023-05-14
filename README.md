# Project Guide

*A guide to setup and run projects*

## Step 1 : Cisco AnyConnect

*In order to run HPC, you must connect to the VPN using Cisco AnyConnect.*

Guide to [SJSU Virtual Private Connection (VPN) Setup](https://www.sjsu.edu/it/services/network/vpn/index.php)

### Download VPN client

1. [Go to the VPN website at vpn.sjsu.edu](https://vpn.sjsu.edu/)
2. Select the "Group" item that matches your role at SJSU
3. Sign in with your SJSU ID and password
4. Follow the on-screen instructions for your device and browser

### Using the VPN client

-  [MacOS [pdf\]](https://www.sjsu.edu/it/docs/connectivity/vpn-feb-2021/SJSU-VPN-Guide-MAC-Students.pdf)

- [Micosoft Windows [pdf\]](https://www.sjsu.edu/it/docs/connectivity/vpn-feb-2021/SJSU-VPN-Guide-WIN-Students.pdf)

##### **Login Credentials to Cisco Anyconnect will be your SJSU credentials**

- Student ID (**NOT** student email)

- Password

  

#### Help

Contact your department [IT Support Tech](https://www.sjsu.edu/it/support/desktop/desktop-support-contacts.php) or contact the [IT Service Desk](https://www.sjsu.edu/it/support/service-desk/index.php)

## Step 2: Login to HPC Cluster

*In order to run jobs, you need to be connected to the [HPC (High Performance Computing) cluster](http://coe-hpc-web.sjsu.edu/). There are two different types of nodes: Login node and Compute node. This is how to access the login node*

### 1. Connect to Cisco AnyConnect

### 2. Access HPC

- Windows

  - Connect via PuTTY to coe-hpc.sjsu.edu. Detailed instructions to do so can be found on the Web, e.g., [here](https://www.ssh.com/academy/ssh/putty/windows).

- Linux/Mac

  - Open the Terminal app and type:

    ```shell
    ssh SJSU_ID@coe-hpc.sjsu.edu
    
    Or ssh SJSU_ID@coe-hpc1.sjsu.edu if the previous command gives you time out error via VPN (The second one typically works for me)
    ```

       - SJSU_ID **must** be replaced with you STUDENT ID (**not** your student email)
       - You will be prompted for your SJSU password. 

## Step 3: Request a Node

*In order to run jobs, you must [Request a Node](http://coe-hpc-web.sjsu.edu/). There are two types of nodes that could be requested: Compute Nofe and GPU Node. There are also two types of requests: `srun` and `sbatch` *

### Compute Node

#### srun

*srun is a way to run interactively by specifying a pseudo-terminal the task should execute in, which is accomplished by using the *–pty* flag*

##### Example:

```` shell
srun --ntasks 1 --nodes 1 --cpus-per-task 4 --pty /bin/bash
--or--
srun -n 1 -N 1 -c 4 --pty /bin/bash
--or--
srun -p gpu --gres=gpu --pty /bin/bash
````

###### *Options* 

- `--ntasks/-n`: specify the number of tasks
- `--nodes/-N`: specify the number of nodes
- `--cpus-per-task/-c`: specify the number of cpus allocated per task
- `--gres=gpu`: used when requesting a GPU resource

## Step 4: Install Conda

*Any packages must be installed in the login node*

1. Download Anaconda package inside the login node

   ````shell
   wget https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh
   ````

2. Install the Anaconda package

   ```` shell
   bash Anaconda3-2022.05-Linux-x86_64.sh
   ````

## Step 5: Create conda Environment

*Creating a conda environment for running projects.*

### Create Environment

*To create a conda environment, use this command: (You can replace **myenv** with anything you want like bioproj or maryenv)*

```` shell
conda create --name ner_biobert
````

​	view  [Manage conda environments](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) to learn more about creating environments

### Activating Enviroments

*Activate your conda environment with the command: (again replacing **myenv** with what you named your environment):*

```` shell
conda activate ner_biobert
````



### Deactivating Environments

*To deactivate your conda environment, use this command:*

```` shell
conda deactivate
````



## Step 6: Install Tools in conda Environment



**In order to install these tools, you must activate you conda environment**

``` bash
conda install pandas
pip install metapub
conda install transformers
pip install simpletransformers
conda install jupyter
conda install git-lfs
git lfs install
```

## Step 7: Setup Biobert

In folder with jupyter notebook file

```bash
git lfs install
git clone https://huggingface.co/dmis-lab/biobert-v1.1
```

## Step 8: Run load_dataset() in the login node

When the conda environment is activated, run an interactive python session by running `python`

an interactive python shell with start. Enter the following code. It will take a few hours to complete but once the abstracts are loaded, they will be stored in your home folders .cache

```python
>>> from datasets import load_dataset
>>> import nltk
>>> nltk.download('punkt')
>>> dataset = load_dataset('pubmed')
```

## Step 9: Run Jupyter Notebook

Open three terminals

1. Terminal 1 (ssh to HPC) (compute node)

```bash
srun --ntasks 1 --nodes 1 -t 24:00:00 --cpus-per-task 55 --pty /bin/bash
```

​	Remember the compute node number

```bash
conda activate [desired conda environment]
jupyter notebook --no-browser --port=8001
```

​	Copy either of the 2 links

2. Terminal 2 (ssh HPC) (login node)

```bash
ssh -N -L 8001:localhost:8001 c[node number from above]
```

3. Git Bash (not connected to HPC)

````bash
ssh -N -L 8001:localhost:8001 [your SJSU ID][@coe-hpc1.sjsu.edu
````

​	Enter password

4. Open link from Terminal 1 in browser

## Step 10: Run final_ner.ipynb

Once the you've opened the link from Step 9, run the jupyter notebook final_ner.ipynb. This is the Named-Entity Recognition part of our project as well as the pre-processing for relation extraction

1. Specify the section of pubmed articles you would like to predict under the **TEST** section. with code:

````python
dataset = data[2500000:2600000]['MedlineCitation.Article.Abstract.AbstractText']
````

2. Specify output number in folders where [#] is

```python
header='index\tsentence\tlabel'
with open('test_large[#].tsv', 'a', newline='') as f_output:
    tsv_output = csv.writer(f_output,quoting=csv.QUOTE_NONE,escapechar='/')
    tsv_output.writerow([header])
  
    for i in range(len(new_sentences)):
        data = f'{i}\t{new_sentences[i]}\tNONE'
        tsv_output.writerow([data])
        
header='gene1\tgene2\tsentence'
with open('names_large[#].tsv', 'a', newline='') as f_output:
    tsv_output = csv.writer(f_output,quoting=csv.QUOTE_NONE,escapechar='/')
    tsv_output.writerow([header])
  
    for i in range(len(related_genes)):
        data = f'{i}\t{related_genes[i]}'
        tsv_output.writerow([data])
```

3. Run final_ner.ipynb
4. copy test_large[x].csv to BioRE-drugprot-kuaz/format_dt_Neg-sentence_customvocab with the following in shell:

```bash
cp test_large[x].csv to BioRE-drugprot-kuaz/format_dt_Neg-sentence_customvocab/test.tsv
```

## Step 11: Run Relation Extraction

1. Go to the Relation Extraction repo:

`cd BioRE-drugprot-kuaz`

2. Edit run.sh to change the epochs or path to the BioBERT mode

``` bash
#!/bin/bash
export SEED=0
export CASE_NUM=`printf %02d $SEED`

export LM_FULL_NAME=./biobert_model/biobert-v1.1
export SEQ_LEN=192
export BATCH_SIZE=16
export LEARN_RATE=2e-5
export EPOCHS=10
export RE_DIR=./format_dt_Neg-sentence_customvocab/
export OUTPUT_DIR=./output/bs-${BATCH_SIZE}_seqLen-${SEQ_LEN}_lr-${LEARN_RATE}_${EPOCHS}epoch_iter-$CASE_NUM
mkdir $OUTPUT_DIR
echo $OUTPUT_DIR

export TASK_NAME=bc7dp
export CUDA_VISIBLE_DEVICES=0

python run_re_hfv4.py --model_name_or_path ${LM_FULL_NAME} --task_name $TASK_NAME --do_train --do_eval --do_predict --train_file $RE_DIR/train.tsv --validation_file $RE_DIR/dev.tsv --test_file $RE_DIR/test.tsv --typeDict_file $RE_DIR/typeDict.json --vocab_add_file $RE_DIR/vocab_add.txt --max_seq_length $SEQ_LEN --per_device_train_batch_size $BATCH_SIZE --per_device_eval_batch_size 512 --learning_rate ${LEARN_RATE} --num_train_epochs ${EPOCHS} --warmup_ratio 0.1 --output_dir $OUTPUT_DIR/ --logging_steps 2000 --eval_steps 2000 --save_steps 10000 --seed $CASE_NUM

```

3. Run the relation extraction:

``` ./run.sh 2>&1 | tee outlog.txt ```

The output will be in the directory: `OUTPUT_DIR=./output/bs-${BATCH_SIZE}_seqLen-${SEQ_LEN}_lr-${LEARN_RATE}_${EPOCHS}epoch_iter-$CASE_NUM`

The predictions will be in the file `predict_results_bc7dp.txt`

## Step 12: Merge output with names

After relation extraction is performed, merge names_large[#].tsv with predict_results_bc7dp.txt to get the names and relationships
