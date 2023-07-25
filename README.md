# Supervised and unsupervised LLM evaluation

This repository describes two techniques for evaluating LLM performance. It also includes some thoughts on an evaluation methodology.

## Supervised

There are several test harnesses for LLM evaluation using canned data. The most well known is [HELM](https://crfm.stanford.edu/helm/latest/). 

EleutherAI produces a simpler harness called [lm-eval](https://github.com/EleutherAI/lm-evaluation-harness). It is fairly easy to run locally and completes one scenario in a few hours on a `g4dn.4xlarge` instance. It supports most HuggingFace models out of the box, and while they haven't documented a way to add a new model, it is relatively easy to do so (see this [pull request](https://github.com/EleutherAI/lm-evaluation-harness/pull/562/files)).

In order to use Eleuther to test a new model, I followed these steps:

* Create a new EC2 instance of type `g4dn.4xlarge` using the Deep Learning AMI.
* Install lm-eval using the instructions on GitHub
* Run a script to run one model on one scenario.

This script evaluate Falcon-7B against the _hellaswag_ benchmark.

    python main.py --model hf-causal --model_args pretrained=tiiuae/falcon-7b,trust_remote_code=True --tasks hellaswag --device cuda:0

For this example, Falcon has an accuracy of 57.8%. The equivalent number for llama-7B is 56.4%.

### HELM 

Running HELM takes a few more steps.  First, deploy a `g4dn.12xlarge` EC2 instance using the Deep Learning AMI on Ubuntu 20.04. We use a powerful EC2 instance because we want to run models locally.

Follow these [instructions](https://www.mongodb.com/docs/manual/tutorial/install-mongodb-on-ubuntu/) to set up a local MongoDB instance to use for caching.

Next set up a Conda environment: 

    conda create -n crfm-helm python=3.8 pip

Activate the environment: 

    conda activate crfm-helm

Clone the [main branch](https://github.com/stanford-crfm/helm) to a local directory. We use the main branch because we need the fix that avoids trying to send data to Google's Perspectives API, and that fix was committed in May 2023. The current released version (0.2.2) dates to March.  Go into the cloned repository and run: 

    pip install .

Create a file called `run_specs.conf` with this line:

    entries: [{description: "boolq:model=stanford-crfm/BioMedLM", priority: 1}]

This tells is to run the `boolq` scenario using a model from HuggingFace called `BioMedLM`. This file can contain multiple entries; see [the example](https://github.com/stanford-crfm/helm/blob/main/src/helm/benchmark/presentation/run_specs.conf) for more details.

Now run the benchmark:

    helm-run --conf-paths run_specs.conf --suite v1 --max-eval-instances 1 --local --mongo-uri mongodb://127.0.0.1:27017/helmdb --enable-huggingface-models stanford-crfm/BioMedLM

Note a few command line options:

* `--mongo-uri` provides the connection string for the Mongo database
* `--enable-huggingface-models` lets us specify any model from the HuggingFace model hub

After the run completes, run this to produce the results:

    helm-summarize --suite v1

And run `helm-server` to get the web UI for browsing the results.

#### Evaluating a local fine-tuned model

If you run a HuggingFace fine-tuning job in SageMaker, you can run HELM against it.

First, download the training model artifact from S3. Expand the artifact into a directory.

    aws s3 cp s3://<model artifact path> .
    mkdir smmodel
    cd smmodel
    tar zxf ../model.tar.gz 

Now create a run configuration that references that model.

    entries: [{description: "mmlu:subject=philosophy,model=huggingface/smmodel", priority: 1}]

Finally, run the evaluation.

    helm-run --conf-paths run_specs.conf --suite v1 --max-eval-instances 1 --local --mongo-uri mongodb://127.0.0.1:27017/helmdb --enable-local-huggingface-models ./smmodel

#### HELM in local mode

Note that HELM normally does not run models locally. It invokes them via an API. For many models you need to provide an API key. If you look at the [model proxy code](https://github.com/stanford-crfm/helm/blob/main/src/helm/proxy/models.py), you can see the list of supported models. For example:

    Model(
        group="together",
        name="databricks/dolly-v2-3b",
        tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG],
    ),

The first half of the model name is the organization, `databricks` in this case. In the [auto client code](https://github.com/stanford-crfm/helm/blob/main/src/helm/proxy/clients/auto_client.py), you can see which model client is used for each organization. For example:

    elif organization in ["together", "databricks", "meta", "stabilityai"]:
        from helm.proxy.clients.together_client import TogetherClient
        client = TogetherClient(api_key=self.credentials.get("togetherApiKey", None), cache_config=cache_config)

Then in the [clients](https://github.com/stanford-crfm/helm/tree/main/src/helm/proxy/clients) folder you can check for the specific client implementation. The [together client](https://github.com/stanford-crfm/helm/blob/main/src/helm/proxy/clients/together_client.py) uses an API key, so you'd have to sign up and provide that before using these models. 

For that reason, our examples so far used the HuggingFace models, as they run locally on the machine.

## Unsupervised

In cases where we don't have labeled ground truth data, we need an unsupervised technique. One option is to use a high quality reference model as an evaluator. The evaluator judges how well two or more other models perform. This technique is inspired by two papers:

* [G-Eval](https://arxiv.org/abs/2303.16634)
* [LMExam](https://lmexam.com/)

The notebook `llm-unsupervised-eval.ipynb` shows a simple example of this approach. It uses Claude as the evaluator to judge the performance of Falcon 40B and Flan T5 XL models. The task is to summarize news articles from `cnn-dailymail` dataset. 

We give the evaluator a prompt that lays out the evaluation criteria, and then collect the output scores for comparison. We also had the evaluator look at the ground truth data as a sanity check.

The assumption in this technique is that the evaluator is as good as a human. That's a complex question, but we can satisfy it to some extent by comparing against ground truth data first, and by asking the model to explain its evaluation. (And why don't we just use the higher quality model? Well, we might want a model that's cheaper or tuned to a specific domain.)

## Evaluation methodology

You may wish to compare the performance of two or more LLMs for several reasons:

* You want to try a smaller, lower-cost model, but are worried about how it performs compared to a larger model.
* You are trying a new task with your own prompts and data, and want to see how a model performs in that specific scenario.
* You have built a new version of a model using transfer learning and want to see if the performance has gone up or down.

Let's walk through the steps you might take when picking a good model for a use case. 

### Starting from scratch

If you are not sure which models to try for a specific use case, start here. These three steps will help you get to a short list of feasible models.

#### Step 1: Narrow based on task

Use resources like the [HuggingFace model hub](https://huggingface.co/models) and [HELM](https://crfm.stanford.edu/helm/latest/?groups=1) to identify models that can perform the task. Also consider the models available through [SageMaker Jumpstart](https://docs.aws.amazon.com/sagemaker/latest/dg/jumpstart-foundation-models-choose.html) and [Amazon Bedrock](https://aws.amazon.com/bedrock/).

#### Step 2: Consider infrastructure constraints

We won't explore this in detail, but consider what type of GPU you'll need to run a the largest models. That will impact cost and performance.

#### Step 3: Narrow based on public benchmarks

Using public benchmarks like HELM or the HuggingFace [leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard), identify which feasible models rank the best in the benchmarks. Note that not all models are listed in those benchmarks. 

### Final selection from a few candidate models

Once you've identified a few models that seem like a good fit, or if you need to evaluate a new model against one you're currently using, you can follow these additional steps.

#### Step 4: Use a supervised benchmark on new model(s)

Run a supervised tool on the new model(s) of interest using canned data.

#### Step 5: Use an unsupervised benchmark on new model(s).

If you need to test against new data or prompts rather than canned data, use an unsupervised tool.

#### Step 6: Incorporate a supervised or unsupervised tool into the MLOps lifecycle

Remember that you need to constantly evaluate the performance of the model you're using. Collect some percentage of the inputs and outputs from your model, and use human or automated evaluation to measure model quality.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.
