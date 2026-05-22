# Hospital Readmission Fairness Audit — Project Plan
**Course:** Algorithmic Fairness, Accountability & Ethics, Spring 2026  
**Dataset:** Diabetes 130-US Hospitals (1999–2008), UCI ML Repository (~101,000 records)  
**Protected attributes:** Race · Gender · Age  
**Models:** Logistic Regression + Random Forest  
**Target:** Readmission within 30 days (binary)

---

## Exam guideline coverage map

| Exam point | Section |
|---|---|
| 1. Problem description | Section 1 |
| 2. Dataset analysis & biases | Section 2 |
| 3. Fairness metrics & justification | Sections 3 & 4 |
| 4. Explainability (whitebox / blackbox) | Section 3 |
| 5. Debiasing the model and/or data | Section 5 |
| 6. Fairness improvement after debiasing | Sections 5 & 6 |
| 7. Critical reflections | Section 7 |
| 8. Ethics & philosophy | Section 8 |

---

## Section 1 — Problem Description
**Exam point 1 | Notebook: `01_problem_framing.ipynb` (markdown cells only)**

- Why hospital readmission prediction matters: CMS Hospital Readmissions Reduction Program (HRRP), financial penalties for excess 30-day readmission rates, value-based care incentives
- Concrete harm scenario: a false negative means a clinician trusts the model's low-risk prediction, discharges the patient without follow-up, and the patient deteriorates at home — for diabetic patients this means undetected ketoacidosis, hypoglycaemic events, or comorbidity progression
- Access audit:
  - Training data? ✓ Publicly available, UCI ML Repository
  - Algorithm? ✓ We build and train it ourselves
  - Predictions? ✓ Full model outputs available
  - Stated purpose: predict 30-day readmission to guide discharge decisions
- Who is affected: Black patients (structural healthcare inequity encoded in labels), elderly patients (clinical complexity suppressed if age removed), uninsured patients (payer_code proxy)
- Research questions:
  1. Does the model perform equally across racial groups?
  2. Does the model perform equally across age groups?
  3. Does bias originate from the data or the modelling?
  4. What happens to accuracy and clinical utility when we debias?
  5. Is there a fundamental fairness–accuracy trade-off, and where does it break down clinically?
- Label construction decision: drop rows where `readmitted = >30` rather than collapsing into the negative class — a patient readmitted after 30 days is a distinct clinical outcome not captured by CMS penalty windows, and collapsing them introduces label noise disproportionately affecting groups with slower access to care

---

## Section 2 — Dataset Analysis & Bias
**Exam point 2 | Notebook: `01_problem_framing.ipynb` ✅ Complete**

- Basic dataset characteristics: shape, dtypes, missing values, recoding `?` as NaN
- Class distribution of readmission by race, gender, age group
- Representation bias: record counts per group — 74% Caucasian, 19% African American, all others under 3%
- Readmission rates by protected attribute — rates are surprisingly similar across racial groups (~17%), meaning the fairness story will emerge from model behaviour (FNR) not raw outcome differences
- Proxy feature analysis: Cramér's V association between `payer_code`, `discharge_disposition_id` and race — confirms payer_code functions as a racial proxy
- Label bias: uninsured patients have lower recorded readmission rates not because they are healthier but because they lack access to return — classic label proxy bias (Obermeyer et al. 2019)
- Age as a clinical signal: hospital stay, number of diagnoses, and medications all increase monotonically with age — age is not merely a protected attribute, it tracks real clinical complexity
- Intersectional analysis: Race × Age — Hispanic elderly patients show dramatically elevated readmission rates (~31% at 90-100); African American patients follow a flatter trajectory across adulthood

---

## Section 3 — Baseline Models & Explainability
**Exam points 3 & 4 | Notebook: `02_baseline_and_explainability.ipynb`**

### 3.1 Model training
- Preprocessing: encode categorical variables, handle missing values, train/test split (stratified by race)
- Train Logistic Regression baseline
- Train Random Forest — differentiator from all 5 example projects
- Report overall accuracy and AUC for both models

### 3.2 Per-group performance
- Use Fairlearn's `MetricFrame` to break down accuracy and AUC by race, gender, and age group
- This is the first signal of where the model fails specific groups

### 3.3 Explainability
- **Logistic Regression (whitebox):** inspect model coefficients directly — which features drive predictions, are any protected attributes or proxies among the top predictors?
- **Random Forest (blackbox):** use SHAP values to explain feature importance and individual predictions — does SHAP reveal `payer_code` or `discharge_disposition_id` as top drivers?
- Compare explanations between the two models
- Address exam question directly: logistic regression is a whitebox model (interpretable by design), random forest is a blackbox model (requires post-hoc explanation methods like SHAP)

---

## Section 4 — Fairness Metrics on Baseline
**Exam point 3 | Notebook: `02_baseline_and_explainability.ipynb`**

Compute the following for both models, broken down by race, gender, and age group:

| Metric | What it measures | Why it matters here |
|---|---|---|
| Demographic parity | Equal positive prediction rates across groups | Baseline check — are some groups flagged high-risk at very different rates? |
| Equal opportunity (FNR) | Equal false negative rates across groups | Most critical — unequal FNR means one group's sick patients are sent home more often |
| Equalized odds | Equal FPR and FNR simultaneously | Stronger criterion — show whether it's achievable and at what accuracy cost |
| Calibration | Predicted probability = actual readmission rate | Does a score of 0.7 mean the same thing for all racial groups? |

- Argue why FNR is the priority metric: a missed high-risk patient is the costliest clinical error
- Establish this as the baseline to compare against after debiasing in Section 5

---

## Section 5 — Debiasing Interventions
**Exam points 5 & 6 | Notebook: `03_debiasing.ipynb`**

Apply at least two of the following three approaches and compare them:

### Pre-processing: reweighting
- Use Fairlearn's `ExponentiatedGradient` or manual sample reweighting
- Upweight underrepresented/disadvantaged groups in training
- Show what changes in per-group FNR

### In-processing: fairness-constrained training
- Fairness-constrained logistic regression via Fairlearn's `ExponentiatedGradient` with equalized odds constraint
- The fairness requirement is baked directly into the optimisation

### Post-processing: threshold adjustment
- Fairlearn's `ThresholdOptimizer` — model trains normally, decision threshold adjusted per demographic group to equalise FNRs
- Least invasive intervention: preserves the learned model, only changes the operating point
- Recommended as primary approach given the clinical argument that the signal in the data (though biased in origin) reflects real health differences

For each intervention:
- Report before/after fairness metrics (demographic parity difference, equalized odds difference, FNR per group)
- Report accuracy cost
- Be honest if an approach does not improve fairness — the exam explicitly asks "does it even improve?"

---

## Section 6 — Fairness–Accuracy Trade-off
**Exam points 5 & 6 | Notebook: `03_debiasing.ipynb`**

- Plot the Pareto frontier: fairness constraint strength on x-axis, accuracy drop on y-axis
- Identify: is there a region where fairness improves cheaply before accuracy collapses?
- Discuss the Chouldechova (2017) impossibility theorem: when base rates differ across groups (which they do here), you cannot simultaneously satisfy demographic parity and equalised odds — show this empirically
- This is the centrepiece of every strong example project

---

## Section 7 — Critical Reflections
**Exam point 7 | Report only**

- Were you able to improve fairness? By how much, and at what cost?
- Should this model exist at all? Does it entrench existing disparities in care quality, or does it reduce them?
- Accountability gap: who is liable when the model is wrong — the hospital, the software vendor, the clinician who relied on it? Neither the EU AI Act nor the US CMS program directly assigns liability for biased predictions
- Structural limits: none of the debiasing interventions fix the upstream problem — African American and uninsured patients have worse health outcomes because of structural inequality in healthcare access, not because of the model
- Recommendations to developers: report per-group FNR as a mandatory model card metric; do not deploy without demographic performance audit
- Recommendations to hospitals: treat model output as decision support, not decision replacement; maintain human oversight especially for disadvantaged groups
- Recommendations to civil society/regulators: mandate algorithmic audits for CMS-deployed models; require transparency on payer_code usage

---

## Section 8 — Ethics & Philosophy
**Exam point 8 | Report only**

### 8.1 Ethical theories applied to dataset bias

- **Utilitarianism:** a utilitarian assessment weighs the aggregate benefit of improved readmission prediction (fewer preventable deteriorations overall) against the harm concentrated in disadvantaged groups from higher FNRs. If the model reduces total readmissions but disproportionately misses Black and uninsured patients, the utilitarian calculus may still approve it — which is precisely the problem with applying aggregate welfare reasoning to healthcare AI
- **Deontology (Kantian):** patients have a right to equal treatment regardless of race or insurance status. A model that produces systematically higher FNRs for one racial group violates that right regardless of its aggregate performance. The debiasing obligation is not contingent on accuracy cost
- **Virtue ethics:** what does a just healthcare system owe its patients? A virtuous discharge coordinator would not defer entirely to a model known to perform unequally; the model's deployment context requires institutional virtue, not just technical fairness

### 8.2 GOFAI to modern AI — was ML necessary here?

- Traditional rule-based clinical decision systems for readmission risk exist (e.g. LACE score, HOSPITAL score) — so this problem is not unsolvable without ML
- The argument for ML: captures complex non-linear interactions between medications, diagnoses, age, and discharge disposition that hand-coded rules miss; scales across 130 hospitals without manual rule maintenance
- The honest answer: it is partly necessary (complexity and scale) and partly pragmatic (faster to train a model than to convene clinical experts to write rules)
- The cost of the pragmatic choice: ML inherits and amplifies historical biases encoded in the training data in ways that rule-based systems — built from clinical expertise rather than historical outcomes — might not

---

## Key papers to cite

| Paper | Where to cite |
|---|---|
| Strack et al. (2014) — Impact of HbA1c Measurement on Hospital Readmission Rates | Dataset description (Section 2) |
| Obermeyer et al. (2019) — Dissecting racial bias in an algorithm used to manage the health of populations | Label bias, ethics (Sections 2, 7) |
| Hardt et al. (2016) — Equality of Opportunity in Supervised Learning | Fairness metrics definition (Section 4) |
| Mehrabi et al. (2021) — A Survey on Bias and Fairness in Machine Learning | Bias taxonomy (Section 2) |
| Chouldechova (2017) — Fair prediction with disparate impact | Impossibility theorem (Section 6) |

---

## Tools & libraries

| Library | Used for |
|---|---|
| `ucimlrepo` | Data loading |
| `pandas`, `numpy` | Data wrangling |
| `matplotlib`, `seaborn` | Visualisation |
| `scikit-learn` | Model training, preprocessing |
| `fairlearn` | Fairness metrics (`MetricFrame`), debiasing (`ExponentiatedGradient`, `ThresholdOptimizer`) |
| `shap` | Blackbox explainability for Random Forest |

---

## Notebooks

| Notebook | Sections covered | Status |
|---|---|---|
| `01_problem_framing.ipynb` | 1, 2 | ✅ Complete |
| `02_baseline_and_explainability.ipynb` | 3, 4 | 🔲 To do |
| `03_debiasing.ipynb` | 5, 6 | 🔲 To do |
