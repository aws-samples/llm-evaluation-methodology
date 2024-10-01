# Evaluate and compare Large Language Models (LLMs) on AWS

As more organizations look to push Generative AI use-cases from initial prototype to production deployment, rigorous testing and evaluation is important to demonstrate robustness, optimize cost, and maximize quality - both before and after initial go-live.

This repository collects some code samples and deployable components that can help you efficiently evaluate and optimize LLM performance - including how to automate testing to evaluate new models and prompt templates faster.


## LLM evaluation overview

***Why?*** ▶️ Systematically measuring and comparing the performance of LLMs (and their configurations like prompt templates) is an important step towards building useful and optimized LLM-based solutions. For example:

- You want to try a smaller, lower-cost model, but are worried about how it performs compared to a larger model.
- You are trying a new task with your own prompts and data, and want to see how a model performs in that specific scenario.
- You'd like to engineer better prompts for your use-case, to achieve the best overall performance with your current model.
- You've built a new version of a model, and want to see if the performance has gone up or down.

***What?*** ▶️ Many factors influence what LLM, prompt templates, and overall solution architecture will be best for a particular use-case, so to inform good decisions your evaluation should consider a range of criteria - like:

- **Usefulness:** Are the answers/completions the model generates *accurate* and *relevant* to what the users need? Is it prone to *hallucinate* or make mistakes?
- **Cost:** Is the cost well-aligned and justifiable for the value the solution will generate? What about other solution-dependent costs like the implementation time and effort, or ongoing maintenance if applicable?
- **Latency:** LLMs require significant computation - how will the model's response speed affect user experience for the use-case?
- **Robustness:** Does the solution give *unbiased*, *stable and predictable* answers? Does it maintain the right tone and handle unexpected topics as you'd like?
- **Safety & Security:** Does the overall solution follow security best-practices? Could malicious users persuade the model to expose sensitive information, violate privacy, or generate toxic or inappropriate responses?

***How?*** ▶️ To address this range of considerations, there's a broad spectrum of evaluation patterns and tools you can apply. For example:

- **Generic vs domain-specific:** Although general-purpose benchmark datasets might give a high-level guide for shortlisting models, task-specific data for your use-case and domain might give very different (and much more relevant) results.
- **Human vs automated:** While human sponsor users provide a 'gold standard' of accuracy for measuring system usefulness, you might be able to iterate much faster and optimize the solution much further by automating evaluation.
- **Supervised vs unsupervised:** Even in 'unsupervised' cases where there's no labelled data or human review, it might be possible to define and measure some quality metrics automatically.
- **LLM-level vs solution-level:** Common solution patterns like [Retrieval-Augmented Generation (RAG)](https://aws.amazon.com/what-is/retrieval-augmented-generation/), [Agents](https://docs.aws.amazon.com/bedrock/latest/userguide/agents.html), and [Guardrails](https://aws.amazon.com/bedrock/guardrails/) combine multiple tools (and perhaps multiple LLM calls) to produce final user-ready responses. LLM call-level evaluations can be useful for optimizing individual steps in these chains, whereas overall solution-level evaluations capture the final end-user experience.


## Getting started

To help get your evaluation strategy up and running, this repository includes:

- A prompt engineering sample app you can deploy in your AWS Account in a region where [Amazon Bedrock](https://aws.amazon.com/bedrock/) is available.
- A deployable [SageMaker Pipeline](https://docs.aws.amazon.com/sagemaker/latest/dg/pipelines-sdk.html) with example configurations for running latency/cost performance tests using [FMBench](https://github.com/aws-samples/foundation-model-benchmarking-tool).
- Some sample notebooks you'll want to run in an [Amazon SageMaker Studio Domain](https://docs.aws.amazon.com/sagemaker/latest/dg/gs-studio-onboard.html) - ideally in the same region for smoothest experience.

▶️ **The simplest way to set up** is by deploying our S3-hosted AWS CloudFormation template (⚠️ Check the *AWS Region* after following the below link, and switch if needed):

[![Launch Stack](https://s3.amazonaws.com/cloudformation-examples/cloudformation-launch-stack.png)](https://console.aws.amazon.com/cloudformation/home?#/stacks/create/review?templateURL=https://ws-assets-prod-iad-r-iad-ed304a55c2ca1aee.s3.us-east-1.amazonaws.com/ab6c96d3-53cf-4730-b0fe-f4762dbbb6eb/cfn_bootstrap.yaml&stackName=LLMEvalBootstrap "Launch Stack")

Alternatively, to guarantee you're in sync with the latest code updates, you can download the template from [infra/cfn_bootstrap.yaml](infra/cfn_bootstrap.yaml) and then [deploy it from the AWS CloudFormation console](https://console.aws.amazon.com/cloudformation/home?#/stacks/create).

> ⚠️ **Note:** The above CloudFormation stacks create an AWS CodeBuild Project with broad IAM permissions to deploy the solution on your behalf. They're not recommended for use in production-environments where [least-privilege principles](https://aws.amazon.com/blogs/security/techniques-for-writing-least-privilege-iam-policies/) should be followed.

If you'd like to **customize** your setup further, check out [infra/README.md](infra/README.md) for details on how to configure and deploy the infrastructure from the [AWS CDK](https://aws.amazon.com/cdk/) source code.


## High-level strategy

Maturing your organization's Generative AI / LLM evaluation strategy is an iterative journey and tooling specifics will vary depending on your use-case(s) and constraints. However, a strong LLM evaluation strategy will typically look something like:

1. **Validate the use-case and architecture:** Without a clear, measurable business benefit case it will be difficult to quantify what good looks like, and decide when to go live or stop investing in marginal improvements. Even if the use-case is important to the business, is it a [good fit](https://aws.amazon.com/generative-ai/use-cases/) for generative LLMs?
2. **Shortlist models:** Identify a shortlist of LLMs that might be a good fit for your architecture and task
    - Curated catalogs like [Amazon Bedrock](https://aws.amazon.com/bedrock/) provide fully-managed, API-based access to a range of leading foundation models at different price points.
    - Broader model hubs like [Amazon SageMaker JumpStart](https://docs.aws.amazon.com/sagemaker/latest/dg/jumpstart-foundation-models-choose.html) and the [Hugging Face Model Hub](https://huggingface.co/models) offer a wide selection with easy paths for deployment on pay-as-you-use Cloud infrastructure.
    - Public leaderboards like [HELM](https://crfm.stanford.edu/helm/latest/?groups=1) and the [Hugging Face Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard) might give useful generic performance indications for models they include - but might be missing some important models or not accurately reflect performance in your specific domain and task.
    - With automatic model evaluations [for Amazon Bedrock](https://docs.aws.amazon.com/bedrock/latest/userguide/model-evaluation.html) and [for Amazon SageMaker](https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-foundation-model-evaluate.html), you can test Bedrock or SageMaker-deployed foundation models with no coding required.
3. **Build task-specific dataset(s) early:** Start collecting reference "test case" datasets for your specific use-case as early as possible in the project, to measure LLM quality in the context of what you're actually trying to do.
    - If your use-case is open-ended like a general-purpose chatbot, try to work with sponsor users to ensure your examples accurately reflect the ways real users will interact with the solution.
    - Collect *both* the most-likely/most-important cases, as well as *edge cases* your solution will need to handle gracefully.
    - If you already have an idea of the internal reasoning to answer each test case, collect that to enable component-level testing. For example record which document & page the answer is derived from for RAG use-cases, or which tools should be called for agents.
    - These datasets can continue to grow and evolve through the project, but will define your baseline for "what good looks like".
4. **Start to optimize:** With reference to task-specific data, run iterative evaluations to narrow your model shortlist and optimize your prompts and configurations.
    - Human evaluation jobs [for Amazon Bedrock](https://docs.aws.amazon.com/bedrock/latest/userguide/model-evaluation-type-human-customer.html) and [for Amazon SageMaker](https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-foundation-model-evaluate-human.html) can help share manual validation work across your internal teams or out to external crowd workers, so you can understand performance and iterate faster.
    - Keep a *holistic* view of performance, accounting for factors like latency, cost, robustness to edge cases, and potential bias - not just accuracy/quality on your target cases.
5. **Automate to accelerate:** From prompt engineering to inference configuration tuning to evaluating newly-released models, there's just too much work to always test by hand.
    - Use automatic evaluation tools to measure model/prompt/solution accuracy metrics across your test datasets automatically: Allowing you to test and iterate faster.
    - *Compare* human and automated evaluations on the same datasets, to *measure* how much trust you can place in automated heuristics aligning with human user preferences.
    - As you accelerate your pace of iteration and optimization, ensure your infrastructure for version control, tracking dashboards, and (re)-deployments are keeping up.
6. **Align automated and human metrics:** With the basics of automated evaluation in place and metrics tracking how well your automated tests align to real human evaluations of LLM output quality, you're ready to consider optimizing your automated metrics themselves.
    - For simple automatic evaluation pipelines, this might be straightforward choices like changing your metric of choice to align to human scores.
    - For pipelines that use LLMs to evaluate the response of other LLMs, this could include prompt engineering or even fine-tuning your evaluator model to align more closely with the collected human feedback.


## Try out the samples

### Data-driven prompt template engineering

Once your `LLMEValWKshpStack` stack has created successfully [in AWS CloudFormation](https://us-east-1.console.aws.amazon.com/cloudformation/home?region=us-east-1#/stacks/), select it from the list and click through to the *Outputs* tab where you should see:

- An `AppDomainName` output with a hyperlink like [***.cloudfront.net]()
- `AppDemoUsername` and `AppDemoPassword` outputs listing the credentials you can use to log in

Open the demo app and log in with the given credentials to get started.

When prompted for a dataset (unless you have your own prepared), upload the sample provided at [datasets/question-answering/qa.manifest.jsonl](datasets/question-answering/qa.manifest.jsonl).

> ⚠️ **Warning:** This sample app is provided to illustrate a data-driven prompt engineering workflow with automated model evaluation. It's not recommended for use with highly sensitive data or in production environments. For more information, see the [infra/README.md](infra/README.md).

You'll be able to:
- Explore the sample dataset by expanding the 'dataset' section
- Adjust the prompt template (noting the placeholders should match the available dataset columns)
- Select a target model and evaluation algorithm, and change the expected reference answer field name, in the left sidebar
- Click 'Start Evaluation' to run an evaluation with the current configuration.

Note that in addition to the default `qa_accuracy` evaluation algorithm from `fmeval`, the app provides a custom `qa_accuracy_by_llm` algorithm that uses Anthropic Claude to evaluate the selected model's response - rather than simple heuristics.

To customize and re-deploy this app, or run the container locally, see the documentation in [infra/README.md](infra/README.md).


### Example notebooks

For users who are familiar with Python and comfortable running code, we provide example notebooks demonstrating other evaluation techniques:

- [LLM-Based Critique (Bedrock and fmeval).ipynb](LLM-Based%20Critique%20(Bedrock%20and%20fmeval).ipynb) demonstrates LLM-judged evaluation for a supervised, in-context question answering use-case, using models on Amazon Bedrock and orchestrating the process via the [open-source `fmeval` library](https://github.com/aws/fmeval).
- [LLM-Based Critique (SageMaker and Claude).ipynb](LLM-Based%20Critique%20(SageMaker%20and%20Claude).ipynb) shows LLM-judged evaluation for a weakly-supervised, text summarization use-case, using Anthropic Claude on Amazon Bedrock to evaluate open-source models deployed on SageMaker.
- [RAG (Bedrock and Ragas).ipynb](RAG%20(Bedrock%20and%20Ragas).ipynb) explores how the open-source [Ragas](https://docs.ragas.io/en/latest/) framework can be used to test integrated Retrieval-Augmented-Generation (RAG) flows with a suite of specialized, LLM-judged evaluation metrics.
- [Conversational Tests.ipynb](conversational-tests/Conversational%20Tests.ipynb) shows how the open-source [agent-evaluation framework](https://awslabs.github.io/agent-evaluation/) can be used to automate testing integrated LLM-based agents/applications against a suite of example customer journeys.

These notebooks have been tested on Amazon SageMaker Studio.


## Further reading and tools

- [FMBench](https://github.com/aws-samples/foundation-model-benchmarking-tool) is an open-source Python package from AWS that can help run performance and cost benchmarking of foundation models deployed on Amazon SageMaker and Amazon Bedrock.


## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the [LICENSE](LICENSE) file. The sample datasets provided in [datasets/question-answering](datasets/question-answering) are transformed subsets of the [Stamford Question Answering Dataset v2.0 dev partition](https://rajpurkar.github.io/SQuAD-explorer/explore/v2.0/dev/) (original available for download [here](https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json)), and are provided under CC-BY-SA-4.0. See the [datasets/question-answering/LICENSE](datasets/question-answering/LICENSE) file.

