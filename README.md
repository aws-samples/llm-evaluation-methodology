# Supervised and unsupervised LLM evaluation

This repository describes two techniques for evaluating LLM performance. It also includes some thoughts on an evaluation methodology.

## Supervised

There are several test harnesses for LLM evaluation using canned data. The most well known is [HELM](https://crfm.stanford.edu/helm/latest/). However, it's a relatively complex job to [add a new model](https://crfm-helm.readthedocs.io/en/latest/adding_new_models/), and the support for HuggingFace models is [brittle](https://github.com/stanford-crfm/helm/issues/1501).

EleutherAI produces a simpler harness called [lm-eval](https://github.com/EleutherAI/lm-evaluation-harness). It is fairly easy to run locally and completes one scenario in a few hours on a `g4dn.4xlarge` instance. It supports most HuggingFace models out of the box, and while they haven't documented a way to add a new model, it is relatively easy to do so (see this [pull request](https://github.com/EleutherAI/lm-evaluation-harness/pull/562/files)).

In order to use Eleuther to test a new model, I followed these steps:

* Create a new EC2 instance of type `g4dn.4xlarge` using the Deep Learning AMI.
* Install lm-eval using the instructions on GitHub
* Run a script to run one model on one scenario.

This script evaluate Falcon-7B against the _hellaswag_ benchmark.

    python main.py --model hf-causal --model_args pretrained=tiiuae/falcon-7b,trust_remote_code=True --tasks hellaswag --device cuda:0

For this example, Falcon has an accuracy of 57.8%. The equivalent number for llama-7B is 56.4%.

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
