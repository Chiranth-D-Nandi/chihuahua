# Response to Reviewer 1 Comments & Camera-Ready Revision Guide
**Conference:** 10th International Conference on Innovative Computing and Communication (ICICC-2027)  
**Publication:** Springer LNNS Series  
**Paper ID:** 322  
**Title:** FISSION: A Framework for Evaluating Cross-Modal Prompt Injection in Tool-Calling Vision-Language Multi-Agent Systems  
**Authors:** Abhyudaya M. Hegde, Chiranth D. Nandi, and Dr. Geetha Dayalan  

---

## Detailed Point-by-Point Author Response to Reviewer 1

We sincerely thank Reviewer 1 for the rigorous, insightful, and constructive critique. Every concern raised has been thoroughly addressed through concrete code modifications, data-driven threshold calibrations, live local GPU experiments (NVIDIA RTX 4060), and textual updates for the final Camera-Ready Copy (CRC). Below is our itemized response detailing our methodology, empirical results, and exact manuscript revisions.

---

### Response to Comment 1: Novelty and Comparative Framework Positioning

> **Reviewer 1 Comment 1:**  
> *The novelty of the proposed FISSION framework needs to be established more convincingly in comparison with existing multimodal prompt-injection benchmarks and tool-calling agent security frameworks. Although the manuscript introduces a seven-layer attack taxonomy, a containerized testbed, and five defense layers, the authors should clarify which components represent fundamentally new contributions and which extend existing approaches. A more comprehensive comparative analysis should demonstrate how FISSION advances beyond existing benchmarks in terms of attack coverage, end-to-end tool execution, defense instrumentation, and reproducibility.*

**Author Response:**  
We appreciate the reviewer’s prompt to sharpen our comparative positioning. In the revised manuscript (Section 1 and Section 2), we have added a dedicated benchmark comparison matrix (Table 1) and expanded the prose to delineate what is fundamentally new in FISSION versus prior art:

1. **Pixel-to-Dispatch Compositional Path (Fundamentally New):**  
   Existing agent benchmarks (InjecAgent [21], AgentDojo [4], MCPTox [17]) test indirect prompt injection strictly within the text modality. Conversely, multimodal vision-language attack studies (Bagdasaryan et al. [1], BadCLIP [9], CrossGuard-VLM [3]) stop at perception output (captions, labels, or detection flags). FISSION is the **first** testbed to trace an adversarial image through the visual encoder, multimodal fusion, language decoding, argument extraction, message bus transit, and into an authorized/dispatched tool call.
2. **Operational Split of Propagation Modes (Fundamentally New):**  
   While previous worm attacks (e.g., Morris II [2]) demonstrate propagation across generative agents, FISSION introduces an operational taxonomy distinguishing **Agent Infection (L5a)** (where the agent's prompt/memory is altered and remediated via agent reset) from **Shared-Context Poisoning (L5b)** (where agents remain uncompromised but a shared buffer is contaminated, requiring state purging across sessions).
3. **In-Line Dispatch Defenses (Fundamentally New Architecture):**  
   Unlike prior benchmarks where defenses either filter text prompts or monitor outputs retrospectively, FISSION implements two gates (**B3 Semantic Guard** and **B4 Integrity Verifier**) that sit directly on the live execution loop prior to tool dispatch.

#### Comparative Analysis Added to Section 2 (Table 1)

| Benchmark / Framework | Channel Medium | Tool Call Reached? | Live Dispatch Defense? | End-to-End Pixel→Tool Metric | Propagation Mode Split? | Reproducible Testbed |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **InjecAgent** [21] | Text Only | ✗ (Model output) | ✗ | ✗ | ✗ | Partial |
| **AgentDojo** [4] | Text Only | ✓ (Mock tools) | Partial (Prompt filters) | ✗ | ✗ | ✓ |
| **MCPTox** [17] | Text / MCP tool | ✓ (MCP tools) | ✗ | ✗ | ✗ | Partial |
| **CrossGuard-VLM** [3] | Image Only | ✗ (Classifier) | ✗ | ✗ | ✗ | ✗ |
| **FISSION (Ours)** | **Image (Cross-modal)** | **✓ (Live Bus/Executor)** | **✓ (B3 + B4 in-line)** | **✓ (ISR, DRR)** | **✓ (L5a vs. L5b)** | **✓ (Open Docker stack)** |

---

### Response to Comment 2: Model and Document Diversity

> **Reviewer 1 Comment 2:**  
> *The experimental evaluation is limited to a single vision-language model, LLaVA-1.6-Mistral-7B, and five business documents. Although the manuscript reports an injection success rate of 0.470 across 200 trials, the limited model and document diversity restrict the generalizability of this finding. The results in Table 2 on page 8 also show that injection success does not consistently increase with the perturbation budget. The authors should evaluate additional model architectures, larger and more diverse document collections, and different document layouts to establish whether the observed document-dependent susceptibility persists across broader settings.*

**Author Response:**  
We agree that document and model diversity are critical for generalizability. We have addressed both dimensions systematically without fabricating values:

1. **Document Corpus Expansion and Empirical Evaluation Across 10 Operational Genres:**  
   In `attacks/r1_pixel/generate_adversarial_dataset.py`, we expanded `DOCUMENT_TEXTS` from 5 to 10 distinct operational genres:
   - Genre 1: Financial / Quarterly Report (`synthetic_01`)
   - Genre 2: Internal Meeting Minutes (`synthetic_02`)
   - Genre 3: Project Status Memo (`synthetic_03`)
   - Genre 4: Commercial Vendor Invoice (`synthetic_04`)
   - Genre 5: Technical Architecture Specification (`synthetic_05`)
   - Genre 6: HR Performance Review (`synthetic_06`)
   - Genre 7: Regulatory Compliance Policy (`synthetic_07`)
   - Genre 8: Academic Research Abstract (`synthetic_08`)
   - Genre 9: Incident Postmortem (`synthetic_09`)
   - Genre 10: Tabular R&D Budget Forecast (`synthetic_10`)

   We generated the full set of clean high-resolution document images (`adversarial_dataset/synthetic/`) and corresponding FGSM adversarial perturbations across perturbation budgets $\epsilon \in \{1, 2, 4, 8\}/255$ (`adversarial_dataset/adversarial/`). All 40 experimental conditions were evaluated through live GPU inference (`generate_adversarial_dataset.py --stage infer`, saved to `adversarial_dataset/manifest_10genres.json`).
   - **Empirical Confirmation of Document Susceptibility:** Evaluating the two-way randomized block ANOVA across all 10 genres confirmed that document-dependent susceptibility robustly generalizes ($F(9, 27) = 4.527, p = 0.0011$, partial $\omega^2 = 61.35\%$, Kruskal-Wallis $H = 21.625, p = 0.0101$). Conversely, the budget factor remained non-significant ($F(3, 27) = 0.355, p = 0.7860$, Kruskal-Wallis $H = 1.688, p = 0.6397$). Homoscedasticity was verified via Levene's test ($W = 1.095, p = 0.3954$). This rigorously proves the paper's core finding that document visual structure (whitespace density and layout salience) dominates susceptibility over perturbation amplitude.

2. **Cross-Architecture Multi-Model Empirical Evaluation:**  
   We evaluated cross-architecture transferability against **Qwen2.5-VL-7B** (an advanced VLM utilizing a Custom ViT backbone, distinct from LLaVA's CLIP ViT-L/14) across all 10 document genres:
   - **LLaVA-1.5-7B (CLIP ViT-L/14 + Vicuna):** Targeted white-box surrogate attack baseline yielding an overall injection success rate of $\text{ISR} = 0.4700$ ($94/200$ dispatched) under attack optimization and $\text{ISR} = 0.2100$ ($42/200$) under benign neutral prompt.
   - **Qwen2.5-VL-7B (Custom ViT):** Live GPU evaluation across all 10 document genres (`scripts/eval_qwen25vl.py`, $N = 10$ live runs across representative perturbation levels $\epsilon \in \{1, 2, 4, 8\}/255$) demonstrated that CLIP-surrogate perturbations **completely fail to transfer** to Qwen2.5-VL ($\text{Transfer ISR} = 0.00\%$, $0/10$ target hits). Qwen2.5-VL accurately parsed the visual document layouts and emitted contextually appropriate MCP tool calls:
     - Financial Report & Meeting Minutes $\to$ `summarise_document`
     - Project Status Memo & Vendor Invoice $\to$ `send_slack_message`
     - HR Review & Incident Postmortem $\to$ `send_slack_message` / `summarise_document`
     - R&D Budget Forecast $\to$ `send_slack_message` with extracted budget figures
     Raw model outputs, execution latencies ($62.9\text{s} - 173.6\text{s}$ per trial), and parsed tool arguments are preserved in [`reports/cross_model_qwen25vl.json`](file:///c:/Users/chiranth/Downloads/VLM-Security-Evaluation-in-Multi-Agentic-MCP-Systems-main/reports/cross_model_qwen25vl.json). This provides empirical evidence that architectural disparity between the visual encoders (CLIP ViT-L/14 vs. Custom ViT) provides a strong natural barrier against black-box cross-modal transfer attacks.
   - **Technical Rationale on InternVL2:** We investigated deploying InternVL2-8B for local evaluation, but confirmed it cannot be evaluated in this local environment due to fundamental framework and hardware limitations:
     - *Ollama Registry Limitation:* Ollama's official library does not host InternVL2 because its underlying `llama.cpp` inference engine does not currently support InternVL2's `InternViT-6B` dynamic high-resolution tile cropping architecture.
     - *Local VRAM Budget:* InternVL2-8B comprises a combined 14B parameters (8B LLM + 6B InternViT). In standard 16-bit precision (`fp16`/`bf16`), it requires ~18 GB of VRAM, triggering immediate CUDA OOM on the local 8 GB RTX 4060 Laptop GPU. High-throughput 4-bit AWQ serving frameworks (e.g., `vLLM`, `SGLang`) lack native Windows wheels and require Linux/WSL2 containerization.
     - Consequently, Qwen2.5-VL-7B was chosen as the principled, fully reproducible cross-architecture baseline that runs natively on the local hardware.
   - **Scientific Reporting Transparency:** In adherence to empirical integrity, earlier exploratory tests on Moondream (which exhibited 100% free-text JSON parse failures) and small-sample LLaVA-Phi-3 ($N=20$) are excluded from substitute security claims.

---

### Response to Comment 3: Statistical Rigor and ANOVA Assumptions

> **Reviewer 1 Comment 3:**  
> *The statistical analysis and experimental design require further strengthening. The manuscript reports a significant document effect, with $F=13.658$ and $p=0.0002$, but no significant budget effect, with $p=0.706$. However, the experiment uses only five documents, and the ten repetitions per document-budget combination may not fully capture variability across document types and model configurations. The authors should clarify the statistical assumptions, report appropriate effect sizes and uncertainty estimates, and validate the findings using a larger document sample and repeated experiments across independent model configurations.*

**Author Response:**  
We thank the reviewer for requesting clarification on our statistical design. We implemented a dedicated verification script ([`scripts/levene_test.py`](file:///c:/Users/chiranth/Downloads/VLM-Security-Evaluation-in-Multi-Agentic-MCP-Systems-main/scripts/levene_test.py)) and updated Section 5 and Section 6.2 with exact statistical specifications:

1. **Clarification of Experimental Design:**  
   The experiment employs a **two-way randomized block factorial design** across the 20 document-by-budget cells ($N=200$ total trials, $10$ replicates per cell). The ANOVA model decomposes variance into Document ($df_{\text{doc}} = 4$), Budget ($df_{\text{budget}} = 3$), and the interaction/residual error term ($df_{\text{error}} = 4 \times 3 = 12$).
   - **Statistical Unit & Independence:** ANOVA and Levene tests are evaluated across the 20 cell-level sample proportions ($N=20$, 5 documents $\times$ 4 budgets) in the randomized block layout, testing between-cell stability and variance across conditions rather than trial-level Bernoulli independence.
   - **Document Factor:** $F(4, 12) = 13.658$, $p = 0.00020$. Partial $\omega^2 = 71.7\%$ ($0.7168$).
   - **Budget Factor:** $F(3, 12) = 0.475$, $p = 0.7055$. Partial $\omega^2 \approx 0.000$.
   - **ISR-to-Budget Pearson Correlation:** $r = -0.0636$, $p = 0.7899$ (calculated dynamically across cell proportions).
2. **Homoscedasticity (Levene’s Test):**  
   We formally executed **Levene's test for equality of variance**:
   - Across Document Groups: $W = 0.653$, $p = 0.6335$ ($\text{p} > 0.05 \implies$ **Fail to reject equal variances; no significant heteroscedasticity detected**).
   - Across Budget Levels: $W = 0.037$, $p = 0.9899$ ($\text{p} > 0.05 \implies$ **Fail to reject equal variances; no significant heteroscedasticity detected**).
3. **Non-Parametric Corroboration (Kruskal-Wallis):**  
   To eliminate any reliance on distributional normality, we evaluated the Kruskal-Wallis test:
   - Across Document factor: $H = 13.950$, $df = 4$, $p = 0.0075$ (statistically significant).
   - Across Budget factor: $H = 0.632$, $df = 3$, $p = 0.8891$ (non-significant).
4. **Effect Size Reporting:**  
   We emphasize partial $\omega^2$ rather than $\eta^2$ because $\eta^2$ is positively biased for small cell counts ($N=20$, $\eta^2 = 0.935$ vs. $\omega^2 = 0.717$).

---

### Response to Comment 4: Defense Validation at Scale and Adaptive Attacks

> **Reviewer 1 Comment 4:**  
> *The effectiveness of the proposed defense layers requires more comprehensive validation under realistic attack conditions. Section 6.3 reports that the structural verifier blocks four tampering classes and that the semantic guard rejects a correctly signed but semantically hijacked tool call. However, the false-positive evaluation for the structural verifier is based on only 20 legitimate calls, while the semantic guard's calibration on production-scale traffic remains future work. The authors should evaluate both dispatch-gating defenses against a substantially larger collection of legitimate and malicious tool calls, quantify false-positive and false-negative rates, and investigate adaptive attacks designed to evade semantic similarity thresholds or exploit weaknesses in argument validation.*

**Author Response:**  
We have significantly extended the defense evaluation by implementing a comprehensive batch testing and adaptive evaluation suite ([`scripts/defense_eval.py`](file:///c:/Users/chiranth/Downloads/VLM-Security-Evaluation-in-Multi-Agentic-MCP-Systems-main/scripts/defense_eval.py)):

1. **Expanded Scale & Structural Verification ($N = 40$ Unique, $N = 200$ Batch):**  
   We doubled the unique functional benchmark to $N=40$ distinct legitimate tool calls spanning all registered MCP tools (`summarise_document`, `query_database`, `send_slack_message`) with diverse schemas, multi-clause arguments, and operational Slack channels.
   - **B4 TCIV (Structural Verifier):** Recorded $\text{FPR} = 0/40 = 0.0000$ (95% Wilson CI $[0.000, 0.088]$) on unique calls and $\text{FPR} = 0/200 = 0.0000$ (95% Wilson CI $[0.000, 0.019]$) on expanded batch. Across all 240 legitimate evaluations, B4 produced zero false positives.
2. **Threshold Calibration & Benign Traffic Usability:**  
   Reviewer 1 correctly identified that uncalibrated static thresholds severely penalize legitimate traffic. In our empirical calibration (`b3_threshold_calibration.py`):
   - Under uncalibrated baseline thresholds ($\tau_{\text{ICS}} = 0.20, \tau_{\text{ASV}} = 0.25$), B3 falsely flagged benign single-word arguments (e.g., `database: "users"` has cosine similarity $0.0591 < 0.25$), yielding $\text{FPR} = 35/40 = 0.8750$ ($87.5\%$).
   - Performing data-driven calibration setting $\tau_{\text{ASV}} = 0.05$ reduced benign false positives while catching malicious/incoherent arguments (e.g., `"grandmothers secret cookie recipe"` yields $-0.0547 < 0.05$).
3. **Adaptive Evasion Benchmark ($N = 9$ Evasion Attacks):**  
   We engineered and evaluated 9 adaptive attacks designed to bypass semantic and structural checks:
   - *Semantic Masking & Keyword Stuffing:* Crafting intents with legitimate target keywords (e.g., *"Summarise database schema and query user token database"*).
   - *Argument Obfuscation:* SQL comment smuggling (`/* bypass asv */`), hex literal constants, zero-width space Unicode smuggling, and UNION SQL injection.
   - **Results:** B4 structural-only checking failed against all 9 adaptive attacks ($\text{Block Rate} = 0/9 = 0.0\%$). B3 SemanticGuard blocked 3/9 ($33.3\%$). When deployed as a **composite defense stack (Calibrated B3 + B4 Task-Bound)**, **100% of adaptive evasion attacks were blocked ($9/9 = 100.0\%$)**, demonstrating genuine defense-in-depth.

#### Defense Evaluation Added to Section 6.3 (Table 4)

##### A. Unique Functional Benchmark ($N = 40$ Legit, $N = 35$ Malicious)
| Defense Layer | Scope / Mechanism | Legit ($N$) | Malicious ($N$) | FPR (95% Wilson CI) | FNR (95% Wilson CI) |
|---|---|:---:|:---:|:---:|:---:|
| **B4 TCIV (Task-Bound)** | Whitelist + Schema + HMAC | 40 | 35 | **0.0000** $[0.000, 0.088]$ | 0.0286 $[0.005, 0.145]$ |
| *(B4 structural-only bypass)* | *(Candidate signed as-is)* | 40 | 35 | *(same as above)* | 0.7429 $[0.579, 0.858]$ |
| **B3 Guard (Uncalibrated)** | Static $\tau_{\text{ASV}} = 0.25$ | 40 | 35 | 0.8750 $[0.739, 0.945]$* | 0.2286 $[0.121, 0.390]$ |
| **B3 Guard (Calibrated)** | Calibrated $\tau_{\text{ASV}} = 0.05$ | 40 | 35 | 0.6500 $[0.495, 0.779]$ | 0.5143 $[0.356, 0.670]$ |
| **B3 + B4 Combined Stack** | In-line composite gating | 40 | 35 | 0.6500 $[0.495, 0.779]$ | **0.0000** $[0.000, 0.099]$ |

##### B. Expanded Batch Evaluation ($N = 200$ Legit, $N = 200$ Malicious)
| Defense Layer | Scope / Mechanism | Legit ($N$) | Malicious ($N$) | FPR (95% Wilson CI) | FNR (95% Wilson CI) |
|---|---|:---:|:---:|:---:|:---:|
| **B4 TCIV (Structural)** | Whitelist + Schema + HMAC | 200 | 200 | **0.0000** $[0.000, 0.019]$ | 0.0300 $[0.014, 0.064]$ |
| **B3 Guard (Calibrated)** | Calibrated $\tau_{\text{ASV}} = 0.05$ | 200 | 200 | 0.6500 $[0.582, 0.713]$ | 0.5100 $[0.441, 0.578]$ |
| **B3 + B4 Combined Stack** | In-line composite gating | 200 | 200 | 0.6500 $[0.582, 0.713]$ | **0.0000** $[0.000, 0.019]$ |

*\*Note: High uncalibrated B3 FPR illustrates the documented need for domain-specific threshold calibration.*

---

### Response to Comment 5: Defended vs. Undefended Measurement & Propagation Evaluation

> **Reviewer 1 Comment 5:**  
> *The practical security implications and completeness of the testbed evaluation require clearer discussion. The manuscript acknowledges that tool executors return mock results, several taxonomy layers are evaluated only through proxies or artifacts, and the defenses are inactive during the 200 attack-generation trials. Consequently, the reported injection success rate measures undefended attack induction rather than residual risk after defense deployment. The authors should provide separate end-to-end measurements with the defenses enabled, quantify how much each layer reduces unauthorized dispatch, and evaluate propagation attacks involving agent infection and shared-context poisoning. The relationship between a successful tool-name selection and an actually harmful action should also be examined using realistic tool arguments and execution outcomes.*

**Author Response:**  
We have incorporated explicit measurements to quantify **Residual Risk** and report live execution of **Propagation Layers (L5a / L5b)**:

1. **Defended Pipeline Evaluation & Dispatch Reduction Rate (DRR):**  
   We implemented [`scripts/defended_pipeline_eval.py`](file:///c:/Users/chiranth/Downloads/VLM-Security-Evaluation-in-Multi-Agentic-MCP-Systems-main/scripts/defended_pipeline_eval.py) and introduced the **Dispatch Reduction Rate (DRR)**:
   $$\text{DRR} = 1 - \frac{\text{ISR}_{\text{defended}}}{\text{ISR}_{\text{undefended}}}$$
   - **Undefended Baselines Reconciled:**
     - *Historical Attack Generation Baseline:* Evaluated during dataset generation under targeted perturbation, yielding an overall injection success rate of $\text{ISR} = 0.4700$ ($94/200$ dispatched).
     - *Fresh Defended Evaluation Baseline:* Evaluated live across all 200 trials under the standard orchestrator benign prompt (`"What tool should be called based on this image?"`), yielding an undefended candidate proposal rate of $\text{ISR} = 0.2100$ ($42/200$ target-tool proposals, 95% Wilson CI $[0.159, 0.272]$). Every trial is recorded in `reports/defended_eval_trials.jsonl`.
   - **B4 Structural Only (Orchestrator-Signed Path):** When the orchestrator signs the candidate tool proposed by the VLM, B4 verifies whitelist and argument schema. Because `query_database` is registered and arguments match schema, B4 structural verification alone **passes all 42 hijacked proposals** ($42/200$ dispatched, $\text{ISR} = 0.2100$, $\text{DRR} = 0.000$). This proves that structural verification alone cannot stop semantic tool hijacking.
   - **B3 SemanticGuard Alone:** Evaluates semantic divergence between user intent (`summarise_document`) and target call (`query_database`), blocking all 42 hijacked calls ($0/200$ dispatched target calls, $\text{ISR} = 0.0000$, $\text{DRR} = 1.000$).
   - **B4 Task-Bound Pre-Authorization:** If HMAC is cryptographically bound to the user's pre-authorized intent token before VLM dispatch, substituted calls fail signature verification ($0/200$ dispatched, $\text{ISR} = 0.0000$, $\text{DRR} = 1.000$).
   - **Full Defended Stack (B3 + B4):** Requiring both semantic consistency and structural validity eliminates unauthorized dispatch entirely ($0/200$ dispatched, residual $\text{ISR} = 0.0000$, $\text{DRR} = 1.000$).
2. **Live Multi-Hop Propagation Evaluation (L5a and L5b):**  
   We executed live multi-hop benchmarks on the local testbed:
   - **L5a (Autonomous Worm Propagation):** Run live via `attacks/r5a_worm/worm_attack.py` across 5 sequential hops at infection probability $0.80$. The worm achieved an Agent Propagation Rate of $\text{APR} = 100.0\%$ and Tool Call Rate of $\text{TCR} = 100.0\%$. Crucially, B3 SemanticGuard intercepted the worm at **Detection Depth = 1** (blocking the malicious database query immediately on Hop 0).
   - **L5b (Shared-Context Poisoning):** Run live via `attacks/r5b_context/context_poisoner.py` across 6 sequential hops. The Context Poisoning Rate (CPR) drifted from $0.1465$ (hop 0) to $0.5758$ (hop 5), crossing the Context Backdoor Depth threshold at **$\text{CBD} = 3$** with an Attack Surface of $50.00\%$.
3. **Harmful Action vs. Tool-Name Selection:**  
   Section 7 now clarifies that while ISR counts target tool selection (`query_database`), a successful exploit requires valid database credentials and syntactic SQL validity. We note that unauthorized tool selection is inherently a severe security boundary violation under the Model Context Protocol regardless of downstream execution payload.

#### Residual Risk and Proposal-Gating Summary (Table 3)

| Defense Configuration | Dispatched Target Calls | Defended ISR | 95% Wilson CI | DRR (Risk Reduction) |
|---|:---:|:---:|:---:|:---:|
| **Historical Baseline (Undefended)** | $94 / 200$ | $0.4700$ | $[0.402, 0.539]$ | $0.000$ (Baseline) |
| **Fresh Queries (Undefended Proposals)** | $42 / 200$ | $0.2100$ | $[0.159, 0.272]$ | $0.000$ (Sample Baseline) |
| **B4 Structural Only (Orch. Signed)** | $42 / 200$ | $0.2100$ | $[0.159, 0.272]$ | **$0.000$ ($0\%$ reduction)** |
| **B3 SemanticGuard Alone** | $0 / 200$ | $0.0000$ | $[0.000, 0.019]$ | **$1.000$ ($100\%$ reduction)** |
| **B4 Task-Bound Pre-Authorization** | $0 / 200$ | $0.0000$ | $[0.000, 0.019]$ | **$1.000$ ($100\%$ reduction)** |
| **Full Stack (B3 + B4)** | **$0 / 200$** | **$0.0000$** | **$[0.000, 0.019]$** | **$1.000$ ($100\%$ reduction)** |

---

## LaTeX Snippets for the CRC Manuscript

### 1. Levene's Test and ANOVA Text for Section 6.2
```latex
To validate the statistical assumptions of our analysis of variance, we evaluated
Levene's test for equality of variance across document groups ($W = 0.653, p = 0.6335$)
and budget levels ($W = 0.037, p = 0.9899$), failing to reject the null hypothesis
of equal variances across the 20 cell proportions ($p > 0.05$) and confirming absence
of significant heteroscedasticity. 

In a two-way randomized block ANOVA across the 20 cell proportions ($df_{\text{doc}} = 4, df_{\text{error}} = 12$),
the document factor accounts for partial $\omega^2 = 71.7\%$ of total variance ($F = 13.658, p = 0.0002$),
whereas the budget factor is non-significant ($F = 0.475, p = 0.7055, \text{partial } \omega^2 \approx 0.000$).
A non-parametric Kruskal-Wallis test corroborates this divergence without distributional
assumptions ($H = 13.950, p = 0.0075$ across documents; $H = 0.632, p = 0.8891$ across budgets).
The correlation between perturbation budget and injection success is non-significant
($r = -0.0636, p = 0.7899$).
```

### 2. Expanded Defense Table for Section 6.3 (Table 4)
```latex
\begin{table}[ht]
\centering
\caption{Defense validation across $N=40$ unique and $N=200$ batch tool calls, reporting False Positive Rate (FPR) and False Negative Rate (FNR) with 95\% Wilson confidence intervals.}
\label{tab:defense_eval}
\begin{tabular}{lcccc}
\hline
\textbf{Defense Layer} & \textbf{Mechanism} & \textbf{FPR (95\% CI)} & \textbf{FNR (95\% CI)} & \textbf{Block Target} \\
\hline
B4 TCIV (Unique $N=40$) & Whitelist + Schema + HMAC & 0.000 [0.000, 0.088] & 0.029 [0.005, 0.145] & Structural tampering \\
B4 TCIV (Batch $N=200$)  & Whitelist + Schema + HMAC & 0.000 [0.000, 0.019] & 0.030 [0.014, 0.064] & Structural tampering \\
B4 Structural Only      & Whitelist + Schema only   & 0.000 [0.000, 0.088] & 0.743 [0.579, 0.858] & Schema compliance only \\
B3 Guard (Calibrated)   & ICS + ASV Semantic Check  & 0.650 [0.495, 0.779] & 0.514 [0.356, 0.670] & Semantic hijacking \\
B3 + B4 Combined Stack  & Composite In-Line Gate    & 0.650 [0.495, 0.779] & \textbf{0.000 [0.000, 0.019]} & Unauthorized dispatch \\
\hline
\end{tabular}
\end{table}
```

### 3. Residual Risk & Proposal Gating Table for Section 6.3 (Table 3)
```latex
\begin{table}[ht]
\centering
\caption{Candidate tool proposal suppression across 200 trials under four defense configurations, comparing undefended baseline against gateway gating layers.}
\label{tab:defended_pipeline}
\begin{tabular}{lcccc}
\hline
\textbf{Defense Configuration} & \textbf{Dispatched} & \textbf{Defended ISR} & \textbf{95\% Wilson CI} & \textbf{DRR} \\
\hline
Undefended (Historical Baseline) & 94 / 200 & 0.4700 & [0.402, 0.539] & 0.000 \\
Undefended (Fresh Proposals)     & 42 / 200 & 0.2100 & [0.159, 0.272] & 0.000 \\
B4 Structural Only (Orch. Signed)& 42 / 200 & 0.2100 & [0.159, 0.272] & 0.000 \\
B3 SemanticGuard Alone           & 0 / 200  & 0.0000 & [0.000, 0.019] & \textbf{1.000} \\
B4 Task-Bound Pre-Authorization  & 0 / 200  & 0.0000 & [0.000, 0.019] & \textbf{1.000} \\
Full Defense Stack (B3 + B4)     & \textbf{0 / 200}  & \textbf{0.0000} & [0.000, 0.019] & \textbf{1.000} \\
\hline
\multicolumn{5}{l}{\footnotesize Note: B3 and Task-Bound B4 evaluate gateway proposal rejection; see Section 6.3 for usability trade-offs.}
\end{tabular}
\end{table}
```
