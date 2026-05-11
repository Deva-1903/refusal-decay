# CS602 Project Report Feedback

This file records all feedback received on project reports throughout the semester.

---

## Report 1 — Full Marks

> Overall, these projects do a fantastic job reflecting both the assignment, and the concepts (e.g., phenomena, mechanisms, etc.) discussed so far in class. Great work.
>
> Each of these proposals appears reasonable for a course project. You should pick the one that, in your opinion, provides the best combination of feasibility and personal interest.
>
> — David Jensen

---

## Report 2 — Full Marks

> The phenomenon is well-explained.

### Assignment description
The goal of this assignment is to clearly describe the research project that you will pursue for the remainder of the semester. In particular, you want to locate that project with respect to the frontier of research in computer science.

- **Select a project.** Based on the feedback that you received on Report 1, and any additional information you have received since writing Report 1, select a specific course project to pursue for the remainder of the semester.
- **Identify and read at least three relevant papers.** Identify and read at least three papers that are the most relevant to the project you have selected. These papers should examine the same or very similar systems, tasks, environments, and phenomena as you have decided to study. If you have not already identified such papers, use online resources (e.g., Google Scholar) and advice from colleagues to select relevant papers. Based on those papers, assess what is currently known about the phenomena you are studying (and the context in which you are studying it). Note that you may need to read more than three papers to identify those most relevant to your work. For background, review the materials from the lectures on Identifying the Frontier (2/24) and Reading Papers (2/26).
- **Update your specification of the project.** Given what you have learned since writing Report 1, briefly describe the system, task, environment, and phenomenon that you are studying. Your reading may have shifted your understanding of these elements since you wrote Report 1 (this is typical). In particular, you may have identified new phenomena or identified new aspects of the system, task, and environment that affect that behavior.
- **Write a brief report.** Write a 3-4 page report with four main sections (below). Figures and tables should be used if needed. Complete bibliographic citations are expected, particularly in the section on "Frontier".
  - **System, task, environment, and implementation** — Briefly describe the system, task, and environment. Describe the specific implementations (e.g., open-source implementations) that are available for you to experiment with or observe. Describe any other infrastructure that exists for experimental or observational study (e.g., data sets, benchmarks).
  - **Phenomena** — Describe one or more computational phenomena that you intend to study in your project. These could be routine types of behaviors (e.g., query processing speed, model test-set performance, error rate) or anomalous behaviors (e.g., system crashes, bottlenecks, etc.).
  - **Variables** — Briefly describe several variables that characterize the system, task, and environment and whose values could possibly affect the phenomenon you are studying. In particular, identify those that you could vary in experiments based on the implementation described above.
  - **Frontier** — Based on your reading to date, describe what is currently known or believed about the phenomenon that you are studying and how it responds to variation in system, task, and environment variables. In particular, describe any gaps in knowledge, weakly supported expectations, or untested assumptions. Please do not merely describe the methods and findings of each paper you read. Instead, synthesize their findings and results to describe the research frontier.

---

## Report 3 — 95/100

> Excellent work. Both surprising findings and great initial work on digging into deeper mechanisms.

### Assignment description
The goal of this assignment is to test your experimental infrastructure and to help hone the focus of your project. Many research hypotheses result from relatively informal observations of a system's behavior. Such an approach can identify unexpected or previously unknown types of phenomena. This type of investigation occupies a middle ground between abstract theorizing (based on prior work) and formal experiments (focusing on a predetermined phenomena). The goal here is exploratory, so we will use a relatively informal approach to generating and analyzing data. The aim here is not a comprehensive set of experiments, but instead a progress report from a preliminary exploration of the space of possible experiments.

- **Identify several expected phenomena.** Briefly describe (in text or graphics) several qualitative and quantitative expectations about how your system will behave in practice. This is exclusively for your own future reference, but it is an important element of good research practice. You cannot be surprised unless you know what you expect.
- **Select a diverse set of specific tasks and environments.** Your goal is to maximize the opportunity for observing unexpected or unusual phenomena, so observe system behavior in a variety of contexts.
- **Determine methods for measuring the occurrence of your expected phenomena.** Record behavior in multiple ways, aiming to measure multiple dimensions or aspects of system behavior.
- **Run the system and analyze the results.** Run a preliminary set of experiments. Either: (a) select one "baseline" point in the multi-dimensional space and vary only one factor at a time from that point; or (b) select a moderate and diverse number of points in the factorial space that you suspect will produce interesting behaviors. Analyze the results using graphics, tables, and summary statistics. Where possible, look at data summaries that show raw data rather than only aggregations.
- **Write a report.** 4-10 page report with sections:
  - **Introduction** — Briefly describe the context (system, task, and environment). Summarize the results of your analysis and how, if at all, the goals and focus of your project have been changed.
  - **Experiments and Results** — Concisely describe the factors that you varied and the ways in which you recorded behavior. Present multiple graphs or tables that summarize the most useful results.
  - **Conclusions and Discussion** — Based on the results, explain your conclusions and what effect they will have on the direction of your project.
  - **Supplementary Materials** — Additional tables, graphs, and details.

---

## Report 4 — 95/100

> A minor point: You state that the RQ1 is mechanistic. I would certainly agree that it is causal, and that the goal of answering the question is to better understand the mechanism behind the effectiveness of prefilling attacks. However, I wouldn't call the question itself mechanistic. Instead, you're asking how one factor (the refusal-direction signal) changes in response to prefilling attacks. Similarly, RQ2 can tell us something about mechanism, but is generally causal (as you state). RQ3 is definitely mechanistic, but the experiments suggested by the hypotheses are mostly causal (and informative about mechanistic hypotheses).
>
> This is all great. You're thinking about mechanism and finding good causal questions to answer that will inform you about mechanism. Nice work.

### Assignment description
The goal of this report is to provide an updated description of your project (system, task, environment, and phenomena) and to state a small number of research questions and hypotheses that you intend to pursue during the remainder of the project.

- **Identify the specific phenomena that you wish to study.** Based on the results of Report 3, select one or more phenomena. Studying a single phenomenon is often enough, though studying several phenomena simultaneously can increase the probability that at least one will produce interesting results.
- **Formulate one or more specific research questions.** Each question should be a single sentence, although there may be more than one question per phenomenon. Research questions should align with the factors you can vary in your experiments and the behaviors you can accurately measure.
- **Formulate multiple hypotheses.** Hypotheses should be falsifiable. Make specific, testable predictions. Explicitly stating and investigating multiple, competing hypotheses is a virtue rather than a vice.
- **Write a report.** 2-4 page report with: (a) an introduction that briefly and clearly describes your current conception of the system, task, environment, and phenomena; (b) one section for each combination of research question and corresponding hypotheses. Use indentation, bold text, or other typographic conventions to make research questions and hypotheses easy to find. Write anonymously for double-blind review.

---

## Report 5 — Not graded

### Assignment description
The goal of this assignment is to produce a plan for your experiments or other analyses so that you can: (a) better anticipate potential pitfalls and resource needs; and (b) get feedback from the instructor and TA on your proposed approach before you encounter potential roadblocks.

- **Choose research question(s) and the hypotheses.** Select the research questions and hypotheses you intend to address. In general, select only a single research question and the hypotheses related to it.
- **Specify the type of empirical analysis you intend to conduct.** Evidence can be derived from experiments, observational studies, simulations, theorems, and combinations of these approaches. Think carefully about what types of evidence would most help evaluate your hypotheses.
- **Specify data collection.** Describe the variables of the algorithm, task, environment, and behavior that will be measured and what units you will use. State which variables you intend to systematically vary and how.
- **Specify the analysis techniques.** Describe the analysis methods that you expect to apply to your data (e.g., visualization methods, summary statistics, linear regression, etc.).
- **Write report.** 4-6 page report with sections: Research questions and hypotheses, Type of analysis, Data collection, Analytic techniques.

---

## Report 6 — Not graded

### Assignment description
The goal of this assignment is to execute your research design and obtain initial results for review by you and the instructor/TA. This will provide the basis for a second iteration of the research design and results in Report 7.

- **Review and update your research design.** Given the feedback received on Report 5, make modifications, additions, or deletions. Consider updates to your research question, hypotheses, data collection, and analysis plans.
- **Collect data.** Execute your research design and collect data to evaluate your hypotheses. If experiments take substantially longer than expected, consider reducing runtime of individual trials, running more trials in parallel (e.g., using Unity), or reducing the total number of trials.
- **Analyze data.** Execute your plan for analyzing the data. Review your hypotheses in light of your results. Consider whether some hypotheses are now sufficiently unlikely. Consider any alternative hypotheses suggested by the results.
- **Write a report.** 4-8 page report including at least: (a) A revised description of the system, task, environment, phenomena, research questions, and hypotheses; (b) A high-level summary of the research design; (c) A detailed description of the design with results; and (d) Conclusions. Where possible, results should be presented in graphs and tables with captions that explain what conclusions should be drawn. The Conclusions section should discuss threats to internal and external validity.

---

## Report 7 — (assignment in progress)

### Assignment description
The goal of this assignment is to revise the research questions, hypotheses, research design, experiments, and conclusions discussed in Report 6 based on any comments received and your own thoughts since writing Report 6.

- **Review and update your research design.** Make any modifications, additions, or deletions that you think necessary. Consider updates to your research question, hypotheses, data collection, and analysis.
- **Collect additional data.** Re-execute key portions of your research design and collect additional data to evaluate your original or revised hypotheses.
- **Perform additional data analysis.** Analyze your additional data and/or revise your analysis of your original data. Review your original or revised hypotheses in light of your results. Consider whether some hypotheses are now sufficiently unlikely that they can be set aside. Consider any alternative hypotheses these results suggest. If your initial analysis raises additional research questions or hypotheses, consider alternative analyses or additional experiments.
- **Write a report.** Revise the text of Report 6 to reflect revised research goals, design, and results. Sections: (a) revised description of system, task, environment, phenomena, research questions, hypotheses; (b) high-level summary of research design; (c) detailed description of design with results; (d) conclusions with threats to internal and external validity.

---

## Recurring themes / lessons learned

- **Causal vs. mechanistic questions** (Report 4): A research question that asks how one factor *changes in response to* another factor is **causal**, not mechanistic — even if its goal is to inform a mechanistic understanding. Mechanistic questions ask *why* / *how* a process works internally; causal questions ask *what effect* a variable has. Frame future RQs accordingly.
- **Strengths to keep doing**: clear phenomenon explanation (R2), surprising findings paired with mechanism-focused follow-up (R3), and good instincts for designing causal experiments that inform mechanistic hypotheses (R4).
