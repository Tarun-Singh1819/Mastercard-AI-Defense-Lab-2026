# Mastercard Innovation Challenge @ GFF 2026
## AI Defense Lab for Payment Security: Attack Taxonomy (Identify Phase)

**Status:** 🟩 **Fully Grounded & Structured**  
**Version:** v2.0 (Executive Release)

---

## 📋 Executive Summary
This document serves as the foundational **Identify (Ideate)** deliverable for the Mastercard Innovation Challenge 2026. It maps the landscape of emerging, Generative AI-powered payment fraud vectors. 

To bridge the gap between ideation, simulation, and detection, this taxonomy categorizes **14 novel attack vectors** into **4 Core Pillars**. Each vector is grounded in real-world technical mechanisms, system vulnerabilities, and outlines specific **Red Team (Simulation)** and **Blue Team (Defense)** strategies to support a closed-loop AI defense system.

---

## 📊 Quick-View Summary Table

| # | Pillar | Vector | Primary Rail / Channel | Grounding Status |
|---|---|---|---|---|
| 1 | **Pillar 2: Social Eng & Impersonation** | Deepfake voice/video impersonation for payment authorization | Wire transfer, corporate treasury, video/call channels | **Grounded** |
| 2 | **Pillar 1: Identity & Onboarding** | Deepfake/synthetic KYC & liveness bypass at onboarding | Account opening, digital KYC | **Grounded** |
| 3 | **Pillar 1: Identity & Onboarding** | Synthetic identity fraud | Credit/lending, account opening | **Grounded** |
| 4 | **Pillar 2: Social Eng & Impersonation** | Adaptive AI scam-conversation agents (voice/chat) | Vishing, scam call centers | **Grounded** |
| 5 | **Pillar 2: Social Eng & Impersonation** | APP scams via AI-personalized social engineering | RTP / UPI / Zelle / FedNow | **Grounded** |
| 6 | **Pillar 2: Social Eng & Impersonation** | AI-personalized Business Email Compromise (BEC) | Corporate wire, ACH | **Grounded** |
| 7 | **Pillar 3: Transaction & Checkout** | GenAI-fabricated dispute/chargeback/refund evidence | Card-not-present, e-commerce disputes | **Grounded** |
| 8 | **Pillar 1: Identity & Onboarding** | Forged financial documents for loan/credit fraud | Lending, account opening | **Grounded** |
| 9 | **Pillar 2: Social Eng & Impersonation** | AI-orchestrated money mule recruitment & network management | Cross-border transfer, P2P, mule accounts | **Grounded** |
| 10 | **Pillar 2: Social Eng & Impersonation** | AI-personalized QR phishing ("quishing") | UPI / QR-based rails | **Grounded** |
| 11 | **Pillar 3: Transaction & Checkout** | Agentic commerce fraud — attacks on AI payment agents | Agent Payments Protocol (AP2), UCP, checkout | **Grounded** |
| 12 | **Pillar 3: Transaction & Checkout** | AI-assisted bot orchestration: card testing, enumeration | Card-not-present, checkout | **Grounded** |
| 13 | **Pillar 4: Evasion & Model Poisoning** | Behavioral-biometric evasion of continuous auth | Session/device auth layer | **Grounded** |
| 14 | **Pillar 4: Evasion & Model Poisoning** | Adversarial poisoning of GenAI fraud-detection models | Cross-cutting (defender's own pipeline) | **Grounded** |

---

## 🏛️ Pillar 1: Identity & Onboarding Fraud

### 2. Deepfake/synthetic KYC & liveness bypass at onboarding
* **Grounded Mechanism:** 
  1. *ID Document Tampering:* Attackers generate entirely synthetic identity document templates or digitally manipulate real IDs using Generative AI (e.g., face-swapping, altering text fields, modifying MRZ/barcode data). These high-fidelity manipulations pass automated OCR and visual authenticity checks.
  2. *Two-Front Biometric Attack:*
     * *Presentation Attack (Sensor Spoofing):* The attacker presents physical artifacts—such as high-resolution printed photos, 3D silicone masks, or video playback screens—directly to the camera to mimic the face matching the ID document.
     * *Digital Injection Attack (Pipeline Bypass):* The attacker bypasses the physical camera sensor entirely. Using virtual cameras (e.g., OBS, custom software drivers), emulators, or rooted/jailbroken devices, they inject pre-recorded deepfake video loops or real-time face-swapping streams directly into the application's media buffer.
  3. *Liveness Evasion:* The injected or presented synthetic video mimics natural micro-movements, depth, or required active challenges (blinking, head rotation, smiling), causing the automated Liveness Detection (Passive and Active Presentation Attack Detection - PAD) to recognize the source as a present, living human.
* **Defense Gaps & Control Failures:**
  * *Sensor Trust Assumption (Pipeline Integrity Gap):* Traditional onboarding apps implicitly trust that incoming media streams originate from the device's physical camera hardware. They lack root/emulator detection, system API hooks checking, and cryptographic verification of the camera pipeline, leaving them vulnerable to digital injection.
  * *Academic vs. Real-world Benchmarking Gap (The "Reality Gap"):* Existing document-tampering and liveness detection models are often trained on limited academic benchmarks. In the wild, they struggle with high-fidelity GenAI-generated documents (e.g., face-swapped IDs as benchmarked in datasets like FantasyID) and advanced, real-time video injection.
  * *Lack of Multimodal Cross-Verification:* Automated onboarding workflows often rely purely on isolated document OCR and biometric matching. They fail to correlate biometric/device metadata (e.g., IP address, system-level emulator presence, webcam plugin activity, OS modifications) with document content and third-party data sources (e.g., credit bureau match, device history) in a real-time risk engine.
* **Closed-Loop Hackathon Strategy:**
  * *Red Team (Generate):* Simulate digital injection attacks by writing scripts that bypass physical camera drivers using Virtual Camera interfaces to feed real-time deepfaced streams during a mock KYC flow.
  * *Blue Team (Defend):* Implement device integrity checks (rooted/emulator checks) and stream signatures to confirm the media originates directly from the camera sensor, paired with visual artifact classifiers to identify face-swap transitions.
* **Reading list:**
  1. [From Forgeries to Foundation Models: A Systematic Survey of Identity Document Attack and Detection](https://arxiv.org/pdf/2607.01442) — Covers both attack categories and detection-side limitations.
  2. FinCEN — [Alert on Fraud Schemes Involving Deepfake Media Targeting Financial Institutions](https://www.fincen.gov/system/files/shared/FinCEN-Alert-DeepFakes-Alert508FINAL.pdf) — Identifies key indicators like webcam plugins, document discrepancies, and virtual environment indicators.
  3. FantasyID dataset paper (arXiv 2507.20808) — Explains digital manipulation ID document datasets and how state-of-the-art forgery detectors fail on in-the-wild manipulations.

### 3. Synthetic identity fraud
* **Grounded Mechanism:**
  1. *Creation (PII Harvesting & Blending):* Attackers harvest legitimate, inactive personal identifiers (specifically Social Security Numbers (SSNs) of "thin-file" demographics like children, the deceased, or homeless people) and blend them with fabricated details (fake names, addresses, birthdays).
  2. *Inadvertent Credit File Establishment:* The attacker applies for a low-scrutiny financial product (retail card, utility service). When the institution queries the credit bureau, the bureau finds no existing record for this combination and automatically generates a **new credit file** for this fabricated person, inadvertently "legitimizing" it in the financial ecosystem.
  3. *Nurturing & Maturation (Building Credibility):* The attacker adds the synthetic identity as an "authorized user" on a legitimate high-limit credit account (piggybacking) to inherit positive credit history, and conducts small utility/credit transactions over months or years.
  4. *Bust-Out Execution:* Once the credit profile matures and score climbs, the attacker applies for large personal loans or credit lines, maxes them out, and disappears. Since there is no "real victim" to report unauthorized charges, the fraud remains undetected, often misclassified as "bad credit debt".
  5. *AI Amplification:* Generative AI acts as an accelerant, automating the production of hyper-realistic supporting documents (payslips, utility bills) and synthetic biometrics (selfies, voices) to bypass digital KYC.
* **Defense Gaps & Control Failures:**
  * *Validation vs. Personhood Verification Gap:* Credit bureaus validate whether data elements (e.g., SSN, address) are *valid* rather than verifying if the combination belongs to a *real, living human*.
  * *Data Fragmentation & Siloed Risk:* Credit bureaus and financial institutions operate with fragmented databases, allowing a synthetic identity to establish small footprints across multiple lenders without any single institution seeing the combined anomalous pattern.
  * *Lack of Graph-Level Relationship Analysis:* Traditional risk engines evaluate credit files in isolation rather than mapping relationships. They fail to use graph neural networks (GNNs) or real-time graph analytics to identify when multiple distinct profiles link back to the same physical address or piggyback off the same source account.
  * *Misclassification as Bad Debt:* Because there is no victim to report the fraud, lenders often write off the losses as credit defaults (bad debt) instead of fraud, hiding the true scale of the synthetic network.
* **Closed-Loop Hackathon Strategy:**
  * *Red Team (Generate):* Generate synthetic user profiles by blending real inactive PII with LLM-generated fake identities and simulate account nurturing behaviors across simulated credit bureaus.
  * *Blue Team (Defend):* Train a Graph Neural Network (GNN) detector to flag identity nodes that share physical attributes (phone number, address clusters) with multiple unrelated accounts, or identify authorized-user "piggybacking" anomalies.
* **Reading list:**
  1. Federal Reserve Bank of Boston — [Gen AI Is Ramping Up the Threat of Synthetic Identity Fraud](https://www.bostonfed.org/news-and-events/news/2025/04/synthetic-identity-fraud-financial-fraud-expanding-because-of-generative-artificial-intelligence.aspx) — Highlights how Gen AI acts as a volatile accelerant to fabricate supporting documents and liveness indicators.
  2. [AI-based Identity Fraud Detection: A Systematic Review](https://arxiv.org/pdf/2501.09239) — Reviews AI-driven detection models and outlines historical performance gaps.
  3. [Neuromorphic graph-analytics engine detecting synthetic-identity fraud](https://wjarr.com/content/neuromorphic-graph-analytics-engine-detecting-synthetic-identity-fraud-real-time) — Documents using real-time graph analytics and transformer-based GNNs to catch structural anomalies in transaction networks.

### 8. Forged financial documents for loan/credit fraud
* **Grounded Mechanism:**
  1. *Layout & Text Generation:* Attackers use generative AI and GANs to create visually and structurally flawless financial documents (pay stubs, bank statements, tax returns).
  2. *Contextual Consistency:* AI models ensure that transaction dates, deposits, tax calculations, and personal information across separate documents are mathematically consistent, evading simple rule-based cross-checks.
  3. *Metadata Scrubbing:* Fraud software automatically strips or fabricates file metadata (EXIF details, PDF tags, author fields) before submission.
* **Defense Gaps & Control Failures:**
  * *Visual/Structural Spoofing of Positional Models:* Automated document verifiers like LayoutLM check joint embeddings of text, visual alignments, and 2D bounding boxes (e.g., verifying total fields align with column borders). Because GenAI replicates these layouts perfectly, the verifier cannot distinguish between a real template and a synthetic one.
  * *Metadata & Origin Blindness:* LayoutLM and similar models evaluate only the *contents* of the image. They are blind to "born-digital" metadata anomalies (e.g., altered timestamps, file creation provenance) and cannot cryptographically verify origin authenticity.
  * *Siloed Database Verifications:* Credit scoring pipelines lack real-time API integrations with payroll providers, tax registries, or bank accounts to perform live, out-of-band verification of the numbers listed on the document.
* **Closed-Loop Hackathon Strategy:**
  * *Red Team (Generate):* Build a generator using Stable Diffusion or LayoutGAN to synthesize fake payslips and bank statements with mathematically consistent text content.
  * *Blue Team (Defend):* Develop a classifier that evaluates digital forensics (compression artifacts, JPEG quantization tables, EXIF metadata mismatches) combined with mathematical cross-checks of document totals.
* **Reading list:**
  1. FBI IC3 — [Criminals Use Generative AI to Facilitate Financial Fraud](https://www.ic3.gov/PSA/2024/PSA241203) — Warns of GAI-enabled document forgery in finance.
  2. [Synthetic Financial Document Generation and Fraud Detection Using Generative AI and Explainable ML](https://www.academia.edu/128755725/Synthetic_Financial_Document_Generation_and_Fraud_Detection_Using_Generative_AI_and_Explainable_ML) — Reviews GAN document generation + LayoutLM classifiers + SHAP explainability.

---

## 🏛️ Pillar 2: Social Engineering & Impersonation

### 1. Deepfake voice/video impersonation for payment authorization
* **Grounded Mechanism:**
  1. *Reconnaissance & Cloning:* The attacker harvests public audio/video recordings of target executives (from YouTube, LinkedIn, webinars) to clone their voices and facial features using Deepfake-as-a-Service tools.
  2. *Telecom Spoofing & Phishing:* The attacker initiates a phishing email or exploits telecom-layer vulnerabilities to perform **Caller ID Spoofing**, making the subsequent phone call look like it originates from a trusted internal line (Group-IB "Voice of Fraud").
  3. *Multi-Participant Video Call:* In the landmark $25.6M (HK$200M) Arup Hong Kong case, the attacker constructed a live multi-person video meeting using real-time video deepfakes of the CFO and other recognizable colleagues to establish high-trust social proof and override the employee's initial skepticism.
  4. *Execution:* The victim, convinced by the visual presence of their team, bypassed secondary authorization lines and executed 15 separate transfers to 5 external accounts.
* **Defense Gaps & Control Failures:**
  * *Over-reliance on Video Verification:* The target employee used the video conference call as a "callback" verification method under the assumption that a video channel is a high-trust, unforgeable medium. Traditional treasury workflows lack mandatory out-of-band verification (e.g., pre-shared passphrases, secondary physical/SMS confirmations) when visual validation is present.
  * *Audio Generalization Gap:* While audio deepfake detectors (Residual CNNs, ResNet-18) perform well on in-domain benchmarks, they suffer from a severe **generalization gap**, losing detection accuracy when encountering new, unseen real-world synthesis methods in the wild.
  * *Academic Misalignment:* Research historically focused on public-figure face-swaps, leaving real-world voice-clone finance vishing under-researched, under-benchmarked, and under-defended ("The Deepfakes We Missed").
* **Closed-Loop Hackathon Strategy:**
  * *Red Team (Generate):* Simulate voice deepfakes by running low-latency TTS models using cloned executive voices to generate fraudulent audio requests.
  * *Blue Team (Defend):* Implement an out-of-band authentication model (SMS OTP/app push) for all wire authorizations, combined with an audio classifier on corporate communication tools to flag deepfake acoustic signatures.
* **Reading list:**
  1. Deloitte — [Deepfake banking fraud risk](https://www.deloitte.com/us/en/insights/industry/financial-services/deepfake-banking-fraud-risk-on-the-rise.html) — Case details and BEC-to-deepfake evolution.
  2. [The Generalization Gap: Do Audio Deepfake Detectors Actually Protect Against Modern Vishing?](https://doi.org/10.3390/electronics15132846) — Shows detection failures due to the generalization gap in unseen synthesis.
  3. Group-IB — [Voice of Fraud](https://www.group-ib.com/resources/research-hub/voice-of-fraud/) — Telecom-layer mechanics (caller ID spoofing combined with cloned voice).
  4. [The Deepfakes We Missed](https://arxiv.org/pdf/2605.12075) — Framework on research misalignment and why voice-cloning vishing is under-defended.

### 4. Adaptive AI scam-conversation agents (voice/chat)
* **Grounded Mechanism:**
  1. *Multi-Turn Script Orchestration:* The attacker deploys an autonomous LLM-based agent (e.g., "ScamAgents") equipped with planning, memory (retaining user context), and speech synthesis/voice cloning APIs to scale voice or chat-based phishing (vishing).
  2. *Dynamic Context-Aware Adaptation:* Rather than using rigid script templates, the agent actively listens to the victim’s responses and dynamically adjusts its persuasion, social proof, or high-pressure tactics in real time to direct the victim towards the target fraud goal.
  3. *Safety Refusal Evasion (Jailbreak Decomposition):* Commercial LLMs have strict system instructions to refuse fraud execution. ScamAgents bypass these guardrails by decomposing the malicious objective into a series of smaller, seemingly benign, task-oriented prompts (e.g., framing the call as a routine security update or an administrative audit). By spreading the instructions across a multi-turn conversation, the agent hides the macro-fraud objective from safety filters.
  4. *Speech-to-Speech Synthesis:* Integrating low-latency voice cloning APIs allows the agent's text outputs to be spoken in real time with human-like pauses and inflection, reducing conversational suspicion.
* **Defense Gaps & Control Failures:**
  * *External Surface Blind Spot:* Scams and social engineering occur on external communication channels (GSM voice, SMS, messaging apps like WhatsApp) that the digital payment system has zero visibility over. Lenders only see the resulting transaction, which appears normal and authorized.
  * *Stateful Multi-Turn Detection Deficit:* Safety guardrails and prompt filters focus primarily on blocking single-turn inputs or static keywords. They lack stateful, multi-turn semantic analysis capable of identifying malicious intent distributed across a series of decomposed, cooperative agent actions.
  * *Lack of Conversational Interception (Interviewer Defenses):* Typical payment applications lack active conversational mitigation frameworks (such as the CASE framework deployed on Google Pay India) that proactively interview users during high-risk transactions to extract the conversational context (e.g., the hook or method of financial loss) and intercept the fraud before the transfer is completed.
  * *Lack of Telecom-Biometric Fusion:* There is no integration between real-time telecom-layer fraud signals (e.g., calls active during payment authorization) and banking applications' fraud detection engines.
* **Closed-Loop Hackathon Strategy:**
  * *Red Team (Generate):* Build an LLM agent that simulates multi-turn social engineering calls by using prompt-decomposition techniques to bypass model refusal guardrails.
  * *Blue Team (Defend):* Implement a Conversational Analyst agent (like Google Pay's CASE framework) that interviews the user during high-risk payment triggers to extract and analyze natural language scam markers.
* **Reading list:**
  1. [A Survey of Agentic AI and Cybersecurity](https://arxiv.org/pdf/2601.05293) — Explains the architecture of ScamAgents, including multi-turn execution and jailbreak decomposition to bypass safety guardrails.
  2. [CASE: An Agentic AI Framework for Enhancing Scam Intelligence in Digital Payments](https://arxiv.org/abs/2508.19932) — Demonstrates a conversational analyst agent deployed on Google Pay India to interview potential victims and extract structured scam intelligence to lift enforcement volumes by 21%.

### 5. Authorized Push Payment (APP) scams via AI-personalized social engineering
* **Grounded Mechanism:**
  1. *Psychological Manipulation:* Attackers exploit human cognitive biases and psychological vulnerabilities—such as trust in authority, fear of loss, or urgency—which are formally categorized under the **Human Vulnerabilities & Exploits (HVE) Framework** (e.g., dual-process theory, visceral state manipulation). Attackers use Generative AI to personalize these scripts at scale using scraped data or voice clones.
  2. *User-Executed Fraud (Authorized Push Payment):* The victim is coerced into initiating and authorizing the transfer (RTP, UPI, Zelle, FedNow) themselves. Because they use their own registered device, correct credentials, and normal location, standard perimeter controls (MFA, IP checks, device fingerprints) see the transaction as legitimate.
  3. *Physical/Cognitive Session Anomalies:* While the security parameters are technically valid, the user's physical interaction with the device during the transaction exhibits anomalies:
     * *Segmented Typing & Hesitation:* Indicating the user is copy-pasting or manually typing unfamiliar destination routing/account details dictated to them in real time.
     * *Device Orientation Shifts:* Picking up, moving, and putting down the phone frequently, indicating they are on an active voice call while transacting.
     * *Extended Session Times:* Caused by following live instructions.
* **Defense Gaps & Control Failures:**
  * *The Authorization Blind Spot:* Standard fraud engines verify *who* is transacting (identity/location check) rather than *under what context* the authorization was obtained. If the real user enters the correct OTP, the payment is processed.
  * *Lack of Continuous Behavioral Biometrics:* Traditional transaction monitoring lacks session-level behavioral intelligence (e.g., mouse micro-movements, typing cadence, phone handling dynamics) capable of detecting coercion, hesitation, or duress in real time.
  * *Lack of Behavioral Vulnerability Mapping:* Risk engines do not incorporate behavioral models like the HVE Framework to assign risk scores to transactions that fit highly-manipulative, high-velocity patterns (e.g., transferring entire savings immediately after registering a new payee).
  * *Active Call/RAT Invisibility:* Banking applications cannot detect if the user is on an active telephone call or if a remote-access-tool (RAT) is running in the background while executing the high-value transaction.
* **Closed-Loop Hackathon Strategy:**
  * *Red Team (Generate):* Simulate coerced payment session telemetry, including delayed typing speeds and simulated device angle tilts to emulate a user being coached over the phone.
  * *Blue Team (Defend):* Implement a behavioral biometric classifier (e.g. BioCatch style) that monitors session-level dynamics (keystroke intervals, touch pressures, gyroscope variations) to detect active coercion.
* **Reading list:**
  1. [The Human Vulnerabilities & Exploits (HVE) Framework](https://arxiv.org/pdf/2606.10083) — Provides a standardized, machine-readable taxonomy for psychological vulnerabilities (like HVSS scoring) exploited by scammers.
  2. Group-IB — [APP Fraud: Global Trends and Tactics](https://www.group-ib.com/resources/knowledge-hub/authorized-push-payment-app-fraud/) — Outlines how APP fraud bypasses standard controls across real-time rails globally.
  3. BioCatch — [How Digital Behavior Can Uncover Social Engineering Scams](https://www.biocatch.com/blog/authorized-push-payment-fraud-how-digital-behavior-can-uncover-social-engineering-scams) — Details how behavioral signals (segmented typing, session length, device handling) reveal when a user is acting under duress.

### 6. AI-personalized Business Email Compromise (BEC) / invoice redirection
* **Grounded Mechanism:**
  1. *Stylometry Mirroring:* Attackers use LLMs to ingest public writing samples (emails, posts, articles) or compromised inbox histories of a target vendor, client, or executive. The AI mimics their specific tone, signature formatting, sentence length, and vocabulary.
  2. *In-Thread Infiltration (BEC 3.0):* Rather than cold phishing, attackers compromise email credentials/session cookies or use lookalike domains to intercept active email threads. They monitor negotiations silently for weeks, and when an invoice is requested, they use GenAI to instantly draft a seamless reply with altered payment routing details that carries the exact styling and context of the ongoing thread.
  3. *Multi-Channel Confirmation:* Fraudsters pair the fake email invoice with a real-time cloned voice call (vishing) or spoofed SMS pretending to confirm the change in wire instructions, providing false multi-channel verification.
* **Defense Gaps & Control Failures:**
  * *Stylometric Detection Bypass:* Traditional Secure Email Gateways (SEGs) look for bad IP reputations, spelling errors, or known malware signatures. They fail against GenAI BEC because the messages have flawless grammar, match the sender's natural tone, and originate from trusted compromised inboxes or high-reputation lookalike domains.
  * *Lack of Context and Relationship Graph Analysis:* Typical corporate setups lack JPMorgan-style email-pattern analytics. These advanced LLM defenses look past the text content to key on:
     * *Relationship Graph Anomalies:* Shifts in communication frequency, timing, or payment routing directions compared to historical patterns between the two entities.
     * *Metadata & Intent Drift:* Analyzing whether the email's semantic intent (e.g., requesting a bank detail change) aligns with historical threads or if it constitutes a sudden departure from the established relationship pattern.
  * *Static Authorization Protocols:* The final payment control relies on a manual callback to a number listed in the email signature (which the attacker controls) rather than verifying details through a pre-registered database or requiring dual-approval for bank-routing edits.
* **Closed-Loop Hackathon Strategy:**
  * *Red Team (Generate):* Build an LLM script that ingests a sample corpus of executive emails and generates stylized replies that request wire changes while mirroring the vocabulary.
  * *Blue Team (Defend):* Implement an LLM classifier that analyzes the relationship graph and flags emails requesting transaction edits that deviate semantically from the history of that sender.
* **Reading list:**
  1. Deloitte — [How generative AI is making fraud easier and cheaper](https://www.deloitte.com/us/en/insights/industry/financial-services/deepfake-banking-fraud-risk-on-the-rise.html) — Documents the rise of the dark-web Crime-as-a-Service ecosystem lowering barriers for scale BEC, predicting US fraud losses to reach $40B by 2027.
  2. [JPMorgan's email-pattern LLM approach](https://paymentscmi.com/insights/ai-artificial-intelligence-use-cases-banking-payments/) — Details defensive LLM models deployed to analyze relationship networks, communication frequency, and behavioral patterns in emails to flag anomalous payment instruction edits.

### 9. AI-orchestrated money mule recruitment & network management
* **Grounded Mechanism:**
  1. *Autonomous Recruitment:* Fraud syndicates use GenAI to automatically generate fake LinkedIn recruiter profiles, launch realistic company websites, and host deepfake video interviewers to recruit unwitting individuals into "remote work" roles.
  2. *Mule Onboarding:* The recruited individuals are instructed to use their personal details to open bank accounts, which are then handed over to the syndicates as money mule accounts.
  3. *Behavioral Blending:* The AI-driven mule orchestration platform manages the transfer of stolen/scammed funds across hundreds of mule accounts. To evade anti-money laundering (AML) detection, it schedules transfers to **mimic** legitimate transaction patterns (e.g., timing transactions during typical business hours, structuring amounts to stay below velocity thresholds, and maintaining low-value transaction history).
* **Defense Gaps & Control Failures:**
  * *Siloed Transaction Monitoring:* Banks monitor accounts in isolation. They fail to share real-time cross-institutional fraud data, allowing mule networks to run funds across multiple banks before AML systems flag the activity.
  * *Static Velocity Rule Evasion:* Traditional monitoring relies on simple, static thresholds (e.g., transaction limits within a 24-hour window). Mule network management engines evade these by dynamically adjusting payment frequency and value below detection triggers.
  * *Lack of Streaming Graph Analytics:* Fraud detectors lack real-time graph database queries to link seemingly separate accounts that share matching device fingerprints, IP networks, login locations, or beneficiary clusters.
* **Closed-Loop Hackathon Strategy:**
  * *Red Team (Generate):* Simulate a multi-node transaction ledger where funds are dynamically structured and passed through mock-mule accounts mimicking normal payroll dispersion distributions.
  * *Blue Team (Defend):* Run real-time streaming graph algorithms (using NetworkX or Neo4j) to detect connected communities sharing matching device hashes, IP subnets, or transaction cycles.
* **Reading list:**
  1. [Mule Networks, Instant Payments, and the New Frontline of Payments Fraud in 2026](https://www.trustsphere.ai/post/mule-networks-instant-payments-and-the-new-frontline-of-payments-fraud-in-2026) — Best mechanism source on the orchestration-platform side.
  2. The Week — [Your job application could make you a money mule](https://www.theweek.in/news/sci-tech/2026/08/04/new-cyber-scam-warning-your-job-application-could-make-you-a-money-mule.html) — Real case study (APG/UNODC report), cross-border logistics detail.

### 10. AI-personalized QR phishing ("quishing") on real-time rails
* **Grounded Mechanism:**
  1. *Tampering & Personalization:* Attackers create physical or digital QR codes and embed them within AI-generated, highly personalized context (e.g., a message stating "Scan this QR to claim your instant refund/cashback").
  2. *Exploitation of Protocol Ambiguity:* The QR code represents a **UPI Collect Request** (a pull transaction). The user scans it believing scanning is a passive or "receiving" action.
  3. *Execution:* The UPI app prompts the user for their PIN. The user enters their PIN, believing it is required to "confirm receipt," but this action technically authorizes a debit (push transaction) from the user's account to the attacker.
* **Defense Gaps & Control Failures:**
  * *Collect vs. Receive UX Ambiguity:* The core protocol allows pull requests ("Collect") to look visually similar to push flows in legacy app versions. The user is vulnerable to social engineering because the UX lacks explicit warnings that PIN entry is *only* for debit actions.
  * *Phasing-Out Lag (P2P Collect Deprecation):* While regulators like the NPCI deprecated P2P Collect Requests to curb fraud, many merchant-category QR systems still allow pull request flows, keeping this exploit surface open.
  * *No Real-Time QR Domain Verification:* UPI applications lack automated checks to verify the safety or risk score of the destination VPA (Virtual Payment Address) encoded in the scanned QR before initiating the transaction.
* **Closed-Loop Hackathon Strategy:**
  * *Red Team (Generate):* Simulate quishing campaigns by generating personalized SMS/WhatsApp pretexts and QR payloads containing mock merchant pull requests.
  * *Blue Team (Defend):* Implement a UPI client-side security wrapper that scans QR destination VPAs against a real-time risk database, prompting warning flags if a P2P collect intent is identified.
* **Reading list:**
  1. [Forensics of UPI and Instant Payment Frauds in India](https://zenodo.org/records/16949256) — Working paper with real case studies (Nagpur, Pune, Hyderabad), good mechanism and forensic detail.
  2. Mondaq/SS Rana — [QR Phishing: A Rough Patch on the Rise of UPI](https://www.mondaq.com/india/white-collar-crime-anti-corruption-fraud/1437116/qr-phishing-a-rough-patch-on-the-rise-of-upi) — Legal/regulatory framing of the rise of UPI-based quishing.

---

## 🏛️ Pillar 3: Transaction & Checkout Exploitation

### 7. GenAI-fabricated dispute, chargeback & refund evidence
* **Grounded Mechanism:**
  1. *Evidence Fabrication (Defect Synthesis):* Fraudsters use image/video-generative models to synthesize highly convincing evidence of "defective" or "damaged" merchandise (e.g., food mold, cracked electronic screens, torn seams).
  2. *Lifecycle Manipulation:* The fraud is executed systematically across four phases:
     * *Transaction Phase:* Setting up the purchase profile to optimize dispute success.
     * *Logistics Phase:* Falsifying shipping weights or intercepting delivery confirmations.
     * *Dispute Phase:* Submitting the AI-synthesized media to claim the item was received damaged.
     * *Communication Phase:* Using LLMs to auto-generate formal dispute arguments that match e-commerce platform policy terminology.
  3. *Exploitation of Proof Assumption:* The attack succeeds by invalidating the system's core trust model—the assumption that digital media represents physical reality.
* **Defense Gaps & Control Failures:**
  * *Lack of Physical-Digital Grounding:* E-commerce systems lack "material anchors" or cryptographic watermarks to prove a photo or video actually matches the specific physical package delivered.
  * *Adversarial Interrogation Deficit:* Merchants attempt "adversarial interrogation" (e.g., requesting real-time, multi-angle videos from specific angles). However, advanced GenAI can easily render matching perspective variations or synthesize full video streams, neutralizing this defense.
  * *Consumer-Centric Policy Bias:* Platforms often operate with a structural bias that automatically favors consumers in "Refund Only" disputes to maintain customer satisfaction, shifting the burden of proof to the merchant.
* **Closed-Loop Hackathon Strategy:**
  * *Red Team (Generate):* Build an image modification script using diffusers to add realistic "cracks" or "scratches" to stock images of electronics.
  * *Blue Team (Defend):* Train a classification model that reviews submitted photos for generative textures, edge anomalies, and consistency across multiple angles.
* **Reading list:**
  1. [Generative AI-Enabled Refund Fraud in Chinese E-Commerce](https://arxiv.org/abs/2606.03215) — Details the 4-phase taxonomy of refund fraud and countermeasure failures.

### 11. Agentic commerce fraud — attacks on AI purchasing/payment agents
* **Grounded Mechanism:**
  1. *Delegated Authority Exploitation:* When AI agents operate with delegated transaction authority (using protocols like Google's AP2 or UCP), attackers target the transaction-mandate lifecycle during runtime execution (e.g., retries or concurrency handling).
  2. *Replay & Context-Binding Attacks:* Attackers intercept valid transaction mandates and replay them, or execute **Context-Redirect Attacks** where they dynamically modify transaction parameters (recipient VPA or pricing) while preserving the signature of the parent authorization.
  3. *Agent Spoofing:* Attackers use forged credentials to impersonate legitimate AI agents to bypass merchant-side checkout rules.
* **Defense Gaps & Control Failures:**
  * *Lack of Zero-Trust Runtime Verification:* Traditional agentic protocols rely on static specification-level validation (signature verification). They lack dynamic context-binding verification and consume-once semantics (such as time-bound nonces) to ensure a mandate cannot be reused or altered during execution.
  * *Absent Intent Cryptographic Binding:* Systems lack trustless intent verification frameworks (like TIVA) that bind persistent decentralized agent identities (DIDs/Verifiable Credentials) to on-chain intent proofs, leaving agent execution environments unverified.
* **Closed-Loop Hackathon Strategy:**
  * *Red Team (Generate):* Build a script that intercepts mock AP2 payment mandates and attempts to replay them to execute secondary unauthorized transactions.
  * *Blue Team (Defend):* Implement a zero-trust runtime verification layer that checks time-bound nonces and enforces cryptographic binding between user intent and agent execution.
* **Reading list:**
  1. [Zero-Trust Runtime Verification for Agentic Payment Protocols](https://arxiv.org/pdf/2602.06345) — eBay paper, documents actual replay and context-binding failures in AP2.
  2. [Secure Autonomous Agent Payments: TIVA](https://arxiv.org/html/2511.15712v1) — Proposes blockchain-based DID intent verification for agent payments.
  3. [How AI agents prevent fraud in autonomous commerce](https://netcoreunbxd.com/blogs/ai-agents-fraud-prevention-autonomous-commerce/) — Frames agent identity spoofing and catalog injections.

### 12. AI-assisted bot orchestration: card testing, enumeration, credential stuffing
* **Grounded Mechanism:**
  1. *AI-Driven Botnets:* Fraud rings orchestrate distributed bot networks that use AI to solve CAPTCHAs and simulate human interaction patterns (like mouse delays and keystrokes).
  2. *Account Enumeration:* The bots systematically cycle through combinations of card details (PANs, CVVs, expiration dates) across merchant checkout forms to discover active card accounts.
  3. *Follow-On Fraud:* Successful card verification is immediately flagged for cash-out or higher-value transactions (Visa reports a 33% follow-on fraud rate within 5 days of a successful enumeration).
* **Defense Gaps & Control Failures:**
  * *Static Rule Limitations:* Traditional security relies on static IP rate-limiting and simple transaction velocity checks. Attackers easily bypass these by spreading requests across thousands of residential IPs (distributed attacks) and executing checks at low frequencies.
  * *Lack of Real-Time Multi-Attribute Scoring:* Standard payment pathways fail to perform sub-20ms multi-attribute scoring (evaluating hundreds of attributes, like Visa's VAAI score) using generative AI trained on global transaction graphs to distinguish normal checkouts from automated enumeration networks.
* **Closed-Loop Hackathon Strategy:**
  * *Red Team (Generate):* Build a simulated bot request script that cycles CVV/PAN variations across a simulated e-commerce gateway, using proxy distribution.
  * *Blue Team (Defend):* Build a real-time risk classifier scoring session-level inputs (card testing patterns, CVV velocity, card type sequence) within a 20ms threshold.
* **Reading list:**
  1. Visa — [VAAI Score announcement](https://investor.visa.com/news/news-details/2024/Visa-Announces-Generative-AI-Powered-Fraud-Solution-to-Combat-Account-Attacks/default.aspx) — Details VAAI Score real-time card testing/enumeration metrics and $1.1B annual losses.

---

## 🏛️ Pillar 4: Evasion & Model Poisoning

### 13. Behavioral-biometric evasion of continuous auth
* **Grounded Mechanism:**
  1. *Behavioral Fingerprint Harvesting:* Attackers record user interactions (mouse tracks, swipe patterns, typing pressure, keystroke dynamics) through malware or malicious SDKs.
  2. *Generative Synthesis (GANs):* Attackers train Generative Adversarial Networks (GANs) on this data. The generator learns to output synthetic interaction patterns that are statistically indistinguishable from the victim's natural behavior.
  3. *Continuous Spoofing Injection:* The attacker injects these GAN-generated behavioral signals during a hijacked session to bypass continuous authentication systems that monitor the session in the background.
* **Defense Gaps & Control Failures:**
  * *Assumption of Behavior Uniqueness:* Current continuous auth layers operate under the assumption that behavioral biometrics are unforgeable and require active human presence. They lack indicators to detect digital injection of interaction streams.
  * *Single-Modal Relyance:* Verification systems check typing or mouse tracks in isolation. They fail to correlate behavioral signals with device-level telemetry (e.g., gyroscope shifts, hardware abstraction layer hooks) that would detect synthetic injection.
* **Closed-Loop Hackathon Strategy:**
  * *Red Team (Generate):* Train a simple GAN on mouse track coordinates to output synthetic trajectories and inject them into a mock session.
  * *Blue Team (Defend):* Implement a multi-modal classifier checking mouse/typing dynamics alongside system telemetry features (e.g., gyroscope and active driver hooks) to catch injected streams.
* **Reading list:**
  1. [Adversarial attacks using GANs on behavioral biometrics](https://arxiv.org/abs/2304.14811) — Reviews GAN-based synthetic mouse movement/keystroke dynamics generation and continuous authentication evasion.

### 14. Adversarial poisoning of GenAI fraud-detection models
* **Grounded Mechanism:**
  1. *Multi-Target Graph Injection (Evasion):* Fraud gangs execute graph injection attacks (such as the MonTi model) to manipulate Graph Neural Network (GNN)-based fraud detectors.
  2. *Node/Edge Spoofing:* Attackers inject crafted malicious nodes (representing fake accounts) and edges (representing normal-looking transactions) into the bank's transaction graph.
  3. *Boundary Shifting:* These injected nodes capture interdependencies between transactions. During the GNN's periodic training cycles, these injected patterns shift the model's decision boundaries, causing actual fraud nodes to be classified as benign.
* **Defense Gaps & Control Failures:**
  * *GNN Training Trust Assumption:* Fraud detection pipelines trust the integrity of input graph datasets. They lack graph-level anomaly detection filters (such as scanning for high-density injected subgraphs or node correlation changes) to prune poisoned nodes before model training.
  * *Black-Box Evasion Vulnerability:* Current detectors are highly susceptible to adversarial graph injections that require no privileged system access (as attackers only need to initiate normal-looking transaction paths to inject nodes/edges).
* **Closed-Loop Hackathon Strategy:**
  * *Red Team (Generate):* Implement a graph injection algorithm to generate and append benign-looking transaction networks to actual fraud nodes inside a transaction graph dataset.
  * *Blue Team (Defend):* Build a pre-training graph defense model that evaluates network density metrics and pruning rules to eliminate poisoned nodes prior to GNN updates.
* **Reading list:**
  1. [Unveiling the Threat of Fraud Gangs to Graph Neural Networks: Multi-Target Graph Injection Attacks Against GNN-Based Fraud Detectors](https://arxiv.org/pdf/2412.18370) — AAAI 2025 paper, documents MonTi (Multi-target one-Time graph injection attack) setting and black-box evasion.

---

## 🌐 Cross-Cutting Enablers
These tooling layers do not constitute independent vectors; rather, they serve as the infrastructural foundation that lowers costs and accelerates the execution of the attacks listed above:
* **Malicious LLMs-as-a-Service:** Systems like FraudGPT, WormGPT, and their active dark-web successors lower the entry barrier, allowing attackers to generate localized, grammatically correct phishing copy, code snippets, and fake personas at scale.
* **General-Purpose Orchestrators:** Re-prompted standard chatbots deployed as agentic handlers to interact directly with victims, managing multiple conversational pipelines in parallel.

---

## 🚪 Out of Scope / Cut Vectors
To maintain a clear and rigorous focus on payment-specific and GenAI-native threats, the following concepts were evaluated and excluded:
* **Insurance Claim Fraud:** Excluded because the core GFF challenge is scoped to payment rails (credit card, wire transfer, real-time rails) rather than property and casualty claims.
* **Generic Malware & Zero-Days:** Excluded because standard endpoint exploitation is a legacy cybersecurity concern and does not specifically exploit GenAI pipelines or payments authorization flows.
* **Insider Threat / Internal Collusion:** Excluded unless a specific GenAI-native vector (such as real-time voice-cloning of supervisors during physical bank vault operations) is established.
