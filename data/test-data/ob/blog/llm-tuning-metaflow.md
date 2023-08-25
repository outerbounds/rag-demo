---
title: "Fine-tuning a Large Language Model using Metaflow, featuring LLaMA and LoRA"
date: "2023-06-21T12:41:26"
tags: ["AI", "Foundation Models", "LLM", "NLP"]
keywords: ["AI", "Foundation Models", "LLM", "NLP"]
categories: ["Foundation Models"]
authors: [federico, obteam]
description: "A workflow template built with Metaflow for fine-tuning LLMs for custom use cases."
image: "/blog/cover/llm-tuning-metaflow.png"
ogImage: "/blog/og/llm-tuning-metaflow.png"
slug: "llm-tuning-metaflow"
---

---

*This post demonstrates how you can use Metaflow to fine-tune a production-grade custom large language model (LLM) based on instruction tuning. We provide an open-source, fine-tuning workflow template that allows you to run models on local or cloud-based GPUs and connect the results to surrounding production systems using the complete toolchain of Metaflow.*

[Federico](https://federicobianchi.io) is a post-doctoral researcher at Stanford University, working on NLP and Large Language Models. He frequently releases research projects as open-source tools that have collectively gathered thousands of [GitHub stars](https://github.com/vinid).

---

Motivation
----------
Unsurprisingly, as of June 2023, building production systems that leverage custom large language models takes a lot of work and money. As usual in any novel technical field, one faces a diverse, quickly moving landscape of immature tooling, missing documentation, and vibrant but chaotic communities. Developers face complex and varying dependencies of LLMs, spanning hardware drivers and bleeding-edge software frameworks, which can take hours and days to figure out for any given use case.

Some astute companies may decide to wait a few years for the field to mature. However, many companies need or want to start innovating faster, and mature tools like [Metaflow](https://metaflow.org/) can help. Metaflow helps you design robust workflows to support any data-intensive application, allowing you to amortize the fixed cost of setting up systems - such as LLM experimentation pipelines - over the project's life. Using Metaflow, teams can set up appropriate human and technical workflows for LLM development and deployment, allowing them to focus on the business's specific needs.

Metaflow has been supporting serious ML use cases like this for years. The foundational needs of LLM development don’t differ much from other earlier models (besides scale!), so Metaflow is a natural fit for building systems around custom LLMs.

Foundational infrastructure for LLMs
------------------------------------
If you just want to run the code and do not care about owning the supply chain or deploying the results to a production system, [Colab](https://colab.google/) and similar GPU-powered-notebook-as-a-service providers are a convenient way to access prototyping the needed infrastructure fast. Alternatively, projects like [GPT4All](https://gpt4all.io/index.html) make running LLM inference on a laptop without a GPU straightforward. 

However, as we discussed in [part one](/blog/custom-llm-tuning) of this series, there are cases where organizations that want to move beyond prototyping in notebooks and consider using LLMs in production need more substantial ML tooling and infrastructure: You need to consider how to connect models to surrounding systems, run training and inference and all supporting steps on a scalable compute platform, iterate and track models, code and data across versions, and have both technical and human workflows in place for continuous improvement and experiments. All this in business-critical systems with tight SLA, security, and compliance requirements.

How Metaflow helps
------------------
This article shows how to use Metaflow to instruction-tune a custom LLM, accompanied by [the full stack of production-grade ML tooling](https://outerbounds.com/docs/infra-stack/) that Metaflow provides. We use a few lines of Metaflow code to access and monitor GPUs in CoreWeave’s cloud, making it easy to produce, track, and report results in shareable [Metaflow cards](https://docs.metaflow.org/api/cards) when running off-the-shelf open-source LLM code.

If you are unfamiliar with the features of Metaflow, we encourage you to look at the [Metaflow documentation](https://docs.metaflow.org). Crucially, all the functionality Metaflow provides can be helpful when developing and deploying real-world systems powered by LLMs.

For instance, Metaflow enables us to explore different model sizes and variants while quickly tuning infrastructure requirements and scheduling runs by changing single values in Python code. Switching between models helps us explore rapidly without unnecessarily wasting money on under-utilized GPUs. 

![](../static/assets/huggingface-instruction-tuning.png)
<Caption> Top left: The lineage of the Alpaca-LoRA model used as a default in this post; Bottom: Steps in the Metaflow workflow; Top right: The output of the fine-tuning step is model checkpoints in HuggingFace  </Caption>

As Metaflow runs any Python code, you can easily leverage state-of-the-art models and APIs from your favorite modeling framework. For instance, In the example workflow, we push checkpoints to the HuggingFace hub with a few lines of code. In our example, we followed the [HuggingFace documentation](https://huggingface.co/docs/transformers/main_classes/trainer#checkpoints) to save all the checkpoints to a local folder, using a specific pattern that is checkpoint-xxx/ where xxx is the number of steps. 

Using HuggingFace APIs with Metaflow makes it easy to create a model repo for each checkpoint, train on whatever hardware you need, and push the results back to the hub to use in an application and for downstream testing. Doing this and sharing incremental model checkpoints on teams can also hugely reduce costs. People need not recompute the same model training results and can instead readily pick up training from where a colleague left off.

Instruction tuning with Metaflow
--------------------------------
For example, let’s fine-tune a recent LLaMA variant, taking an already trained LLaMA language model and training it further on an instruction tuning dataset. The learning task is still language modeling, but the training dataset contains specific instruction-following examples.

In this case, the model learns to do language modeling relevant to the instructions in the Alpaca dataset. You can imagine using any dataset, such as how the Databricks team created a custom instruction-tuning [dataset](https://huggingface.co/datasets/databricks/databricks-dolly-15k) in [Dolly v2](https://www.databricks.com/blog/2023/04/12/dolly-first-open-commercially-viable-instruction-tuned-llm). You can find additional information about Metaflow and large language models in our [blog post on tuning Dolly](https://outerbounds.com/blog/train-dolly-metaflow/).

### Show me the code
The core concepts of this workflow template are the following:
1. HuggingFace to access the upstream models and to checkpoint our fine-tuned models,
2. CoreWeave to access GPU resources, and
3. Metaflow to structure the Python code, run it on the GPUs (carefully monitoring resource utilization with a Metaflow card), and track versions of code and models.

![](../static/assets/coreweave-instruction-tuning.png)

The following code shows how to call the main fine-tuning logic from the [Alpaca LoRA codebase](https://github.com/tloen/alpaca-lora). 

<CustomAdmonition title="Instruction tuning with Metaflow" link="https://github.com/outerbounds/metaflow-instruction-tuning" color="#F9F7F5" image="/icons/common/github.png">
    Use PyTorch, HuggingFace and Metaflow to build a workflow for fine-tuning LLMs.
</CustomAdmonition>

```python
# Contents of the flow.py file in the repository linked above.

from metaflow import FlowSpec, step, Parameter, resources, environment
from mixins import HuggingFaceLora, N_GPU, visible_devices
from custom_decorators import pip, gpu_profile
import os

class LlamaInstructionTuning(FlowSpec, HuggingFaceLora):

    push_checkpoints = Parameter("-push", help="push checkpoints on huggingface", default=False, type=bool)

    @step
    def start(self):
        self.next(self.finetune)

    @environment(vars={
        "CUDA_VISIBLE_DEVICES": visible_devices, 
        "WORLD_SIZE": N_GPU,
        "HUGGINGFACE_TOKEN": os.environ["HUGGINGFACE_TOKEN"],
        "HF_ORGANIZATION": os.environ["HF_ORGANIZATION"]
    })
    @gpu_profile(interval=1)
    @pip(file="requirements.txt")
    @resources(gpu=N_GPU, cpu=16, memory=128000) # tested with A100 and A6000 GPU.
    @step
    def finetune(self):
        self.run()
        if self.push_checkpoints:
            self.upload_to_huggingface()
        self.next(self.end)

    @step
    def end(self):
        pass

if __name__ == '__main__':
    LlamaInstructionTuning()
```

[Metaflow](https://metaflow.org/) provides [a scaffolding for data science workflows](https://docs.metaflow.org/metaflow/introduction), all written in Python. In this case, the flow centers around the `finetune` step, where we use the [multiple inheritance pattern](https://realpython.com/lessons/multiple-inheritance-python/) to modularize the workflow, separating the Alpaca LoRA code that makes HuggingFace API calls from the Metaflow code that organizes the workflow.

Specifically, in the `HuggingFaceLora` class, we define the logic to run the open-source LLM fine-tuning code in a [distributed data-parallel](https://pytorch.org/docs/stable/notes/ddp.html?highlight=distributed+data+parallel) manner using the [torchrun](https://pytorch.org/docs/stable/elastic/run.html) API. The LLM code is run inside a Metaflow task, and a data scientist can add single lines of Python code to layer complex functionality on each task, [like requesting precise GPU and memory resources on the cloud instance that runs the job](https://docs.metaflow.org/scaling/remote-tasks/introduction) and monitoring the resources to ensure they are appropriately sized. 

The key idea for data science teams is that once they have access to a Metaflow deployment, a data scientist who can feasibly write Python code can [write, run, and deploy production-grade workflows in a single work session](https://docs.metaflow.org/introduction/what-is-metaflow). Metaflow [persists and tracks all artifacts automatically](https://docs.metaflow.org/metaflow/basics#artifacts), [exposes them to notebooks](https://docs.metaflow.org/metaflow/client) and other programmatic use cases, and shows them on a UI. Once everything works locally, the workflow can be [deployed to production, connecting to other workflows both upstream and downstream](https://docs.metaflow.org/production/introduction), with a single click.

If you are curious to test all these features in action, you can [do it easily in the Metaflow sandbox](https://outerbounds.com/sandbox). 

Models
------
We fine-tuned four of the recent LLaMA models on the same dataset with a fixed computing budget for each model; we used [Low-Rank Adaptation](https://arxiv.org/abs/2106.09685), making use of the recent [Alpaca LoRA repository](https://github.com/tloen/alpaca-lora). The models we fine-tuned are the 7B, 13B, 33B, and 65B parameters models, with the idea that larger models should provide better performance and answers. We saved multiple checkpoints for each model and chose the best one by validation loss after training. 

We chose the Alpaca dataset - built by generating instructions from GPT-3. We use a [ version](https://huggingface.co/datasets/yahma/alpaca-cleaned) of the dataset in which some errors are corrected. The dataset contains 50K instructions. You can find more details about how we used Metaflow in the practical considerations section.

### Quantitative evaluation
We wanted to assess if our fine-tuned models are still performing as expected after fine-tuning. While there is still active discussion about how to evaluate instruction and chat models (see the [ChatBot Arena](https://lmsys.org/blog/2023-05-03-arena/), for example), we decided to evaluate our models against standard LLM benchmarks as a more customized evaluation approach is beyond the scope of this post. Specifically, we make use of the [LM-Harness](https://github.com/EleutherAI/lm-evaluation-harness) package developed by EleutherAI to evaluate our fine-tuned models on these different tasks in a zero-shot manner. We chose three question-answering benchmarks: [PIQA](https://huggingface.co/datasets/piqa) and [BoolQ](https://huggingface.co/datasets/boolq), and [OpenBookQA](https://huggingface.co/datasets/openbookqa). 

Here is an example from PIQA (the other datasets are similar):

> “To separate egg whites from the yolk using a water bottle, you should…”

The dataset has two possible answers: 

> “Squeeze the water bottle and press it against the yolk. Release, which creates suction and lifts the yolk.”

> “Place the water bottle and press it against the yolk. Keep pushing, which creates suction and lifts the yolk.” 

Given that this is a multiple choice question task, [LM-Harness](https://github.com/EleutherAI/lm-evaluation-harness/blob/big-refactor/docs/new_task_guide.md#multiple-choice-format) runs the evaluation by comparing the likelihood of the label words.

So, how well do these models work? Let’s look at some results (the evaluation metric here is accuracy).

<div style={{alignItems: 'center', justifyContent: 'center', display: 'flex'}}>
<div style={{textAlign: 'center'}}>

| Model | PIQA | BOOLQ | OPENBOOKQA |
| ----- | ---- | ----- | ---------- |
| LLAMA 7B | 0.7916 | 0.7511 | 0.4440 |
| ALPACA 7B | 0.8030  | 0.7835  | 0.4420 |
| LLAMA 13B | 0.8003 | 0.7798 | 0.4480 |
| ALPACA 13B | 0.8030 | 0.8150 | 0.4620 |

</div>
</div>

These results seem coherent with what was expected based on various leaderboards (see [LLaMA results](https://paperswithcode.com/paper/llama-open-and-efficient-foundation-language-1), for example) and papers. There is not a notably high increase in performance, as we expected. 13B parameters models do slightly better than their 7B counterparts.  We do not expect a substantial increase in performance with respect to the original LLaMA models; instead, we would like to maintain similar performances on these benchmarks. Instruction tuning should give us models that follow instructions better and do not perform worse on general benchmarks. Our evaluation shows we have nicely performing models on the latter, but do they follow instructions?

### Qualitative evaluation
When working with LLMs, being mindful that the research field is new and changing rapidly is essential. Therefore, when considering using this technology in a production environment, we need to evaluate qualitatively and measure performance against quantitative benchmarks, which may need significant iteration and revision for assessing performance in a way that makes sense in any given production context.

Even with the small training dataset we used and only three epochs of training, our models already see exciting results. The bigger the model is, the better (and more valuable) its generations become.

<div style={{alignItems: 'center', justifyContent: 'center', display: 'flex'}}>

| Instruction | 7B | 30B |
| ----------- | -- | --- |
| **Write a poem about capybaras** | Capybaras, the largest rodent in the world, <br></br><br></br> With their shaggy coats and <span style={{color: 'red'}}> bushy tails</span>, <br></br><br></br> They roam the grasslands of South America, <br></br><br></br> In search of food and water to quench their thirst. <br></br><br></br> Capybaras are herbivores, and they love to graze, <br></br><br></br> Eating grasses, fruits, and vegetables, too, <br></br><br></br> They are social creatures, living in herds, <br></br><br></br> And they're known for their playful and friendly deeds. | A Capybara's Life <br></br><br></br> A capybara's life is filled with joy, <br></br> As they frolic in the grass and play in the sun. <br></br><br></br> They're the largest rodents in the world, <br></br><br></br> <span style={{color: 'green'}}>With bodies built for swimming and for fun.</span> <br></br><br></br> They're social creatures, always in a group,<br></br><br></br> Enjoying each other's company, without a flop. <br></br><br></br> They love to eat grass and aquatic plants,<br></br><br></br> A diet that keeps them healthy and well-fed. |
| **Can Geoffrey Hinton have a conversation with George Washington? Give the rationale before answering.** (From Flan) | No, it is not possible for Geoffrey Hinton to have a conversation with George Washington. This is because George Washington is a historical figure who lived from 1732 to 1799, while Geoffrey Hinton is a living person who was born in <span style={{color: 'red'}}>1957</span>. Time travel is not possible, so it is not possible for two people from different eras to have a conversation with each other. | No, it is not possible for Geoffrey Hinton to have a conversation with George Washington. Geoffrey Hinton is a living person who was born in 1947, while George Washington died in 1799. It is not possible for a living person to have a conversation with someone who has been dead for more than 200 years. |
| **Classify this document in one of these categories, "sport", "politics", or "technology".** <br></br><br></br> **“The LLaMA models are a family of Natural Language Processing LLMs that can be applied to many different tasks.”** | technology | technology |
| **Do aliens invest in stocks?** | No, aliens do not invest in stocks. Stock investing is a human activity that involves buying and selling shares of publicly traded companies in order to profit from changes in their share prices. Aliens do not have access to the same financial markets and systems that humans do, so they would not be able to participate in stock investing. <br></br><br></br> <span style={{color: 'orange'}}>However, it is possible that aliens could invest in other types of financial instruments</span>, such as bonds, mutual funds, or real estate, if they had access to the appropriate financial markets and systems. | I'm sorry, but as an AI language model, I don't have access to any information about aliens or their investment habits. It's possible that some alien civilizations may invest in stocks, but it's also possible that they may have different investment strategies or may not invest in stocks at all. It's difficult for me to give a definitive answer to this question without more information or data. It's possible that some alien civilizations may invest in stocks, but it's also possible that they may have different investment strategies. |
| **To separate egg whites from the yolk using a water bottle, you should…** (from PIQA) | To separate egg whites from the yolk using a water bottle, you should follow these steps: <br></br><br></br> 1. Crack the egg into a small bowl and separate the yolk from the egg white. <br></br> <span style={{color: 'red'}}>2. Fill the water bottle with cold water. </span><br></br> 3. Place the egg white into the water bottle. <br></br> <span style={{color: 'red'}}>4. Shake the water bottle vigorously to separate the egg white from the yolk. </span><br></br> 5. Remove the egg white from the water bottle and discard the yolk. <br></br> 6. Repeat the process for the remaining eggs. | To separate egg whites from the yolk using a water bottle, you should follow these steps: <br></br><br></br> 1. Start by cracking an egg into a small bowl or cup. <br></br> 2. Hold the water bottle upside down over the bowl or cup. <br></br> 3. Slowly squeeze the water bottle, allowing the <span style={{color: 'red'}}>water to flow</span> into the bowl or cup. <br></br> 4. As the water flows into the bowl or cup, it will push the egg yolk towards the bottom of the bowl or cup. |
| **Best way to pierce ears** (from PIQA)| The best way to pierce ears is to use a sterilized, hygienic piercing gun or needle. It is important to follow the instructions provided by the piercing professional and to clean the area thoroughly before and after the piercing. It is also important to use a high-quality, non-allergenic jewelry and to take proper care of the piercing by cleaning it regularly and avoiding contact with water while the piercing is healing. It is recommended to wait at least 6-8 weeks for the piercing to heal completely before changing the jewelry. | <span style={{color: 'green'}}>The safest and most effective way to pierce your ears is to have it done by a professional piercer at a reputable piercing studio.</span> A professional piercer will use sterilized equipment and follow strict hygienic procedures to minimize the risk of infection or other complications. They will also be able to advise you on the best placement and type of jewelry for your piercing, as well as provide aftercare instructions to ensure proper healing. |
| **What's the best way to drink water if you are juggling three balls?** | If you are juggling three balls, the best way to drink water is to use a straw. This will allow you to keep your hands free to continue juggling, while also ensuring that you don't spill any water on yourself or your juggling equipment. Additionally, it's important to drink small sips of water, rather than gulping down large amounts at once. This will help you to stay hydrated without interfering with your juggling routine. | <span style={{color: 'green'}}>If you are juggling three balls, the best way to drink water would be to take a break from juggling, set the balls down on a safe surface, and take a sip of water from a cup or bottle.</span> Juggling and drinking at the same time can be difficult and may result in spilling water or dropping the balls. It's important to stay hydrated while juggling, but it's best to do so in a way that doesn't interfere with your juggling performance. |

</div>

These results show multiple things: first off, the models are good at replying to questions and instructions. They all come up with coherent text that appears correct. The 7B model is more prone to generating incorrect stuff, such as getting the wrong year of birth of someone or suggesting something that might be unsafe. We have color coded some replies to highlight possible good (green), wrong (red), or unclear (orange) answers.

Summary
-------
You can use the above Metaflow workflow as a generic template for fine-tuning models available on HuggingFace. Besides the Alpaca instruction dataset we used to fine-tune various sizes of LLMs, you can apply the same approach to fine-tune models with your instructions to optimize models for your use cases. We are happy to help you get started with this template and Metaflow in general. Join us and thousands of other ML engineers, AI enthusiasts, and data scientists on  [Metaflow Community Slack](http://slack.outerbounds.co/)! 

Acknowledgments
---------------
We would like to thank Mert Yuksekgonul, Mirac Suzgun, Patrick John Chia, Silvia Terragni, Giuseppe Attanasio, and Jacopo Tagliabue for feedback and suggestions on a previous version of this blog post.

