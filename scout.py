Here’s the combined table — team structure plus steady-state workflow.

⸻

Team Composition and Responsibilities

Role	Core Responsibilities	Steady-State Workload	Notes
Quant (Lead Architect)	System design, retrieval metrics, latency and cost optimisation, SQL/RAG logic	~20–30% ongoing	Oversees architecture, validates technical trade-offs, manages scaling decisions
Data Scientist 1	LLM prompt engineering, agent logic, evaluation scripts	Full-time	Owns /chat and /agent behaviour tuning
Data Scientist 2	Embedding pipeline, vector quality analysis, retrieval QA	Full-time	Monitors recall@k, chunking efficiency
Data Scientist 3	Model evaluation, drift detection, benchmarking	Full-time	Maintains validation datasets and regression tests
Senior Software Engineer	Backend orchestration, FastAPI/Redis/MS SQL integration, deployment automation	Full-time	Responsible for reliability, scaling, and CI/CD
Junior Software Engineer 1	API endpoints, file ingestion, Redis vector operations	Full-time	Implements weekly ETL + embedding jobs
Junior Software Engineer 2	Frontend/chat UI, integration testing	Full-time	Maintains user interface and API client layer
Junior Software Engineer 3	Monitoring, logging, and observability dashboards	Full-time	Prometheus, Grafana, Langfuse setup


⸻

Steady-State Operational Workflow

Area	Example Activities	Frequency
Data Ingestion & Embedding	Weekly ETL → clean → chunk → embed → upsert to Redis Vector	Weekly
Model Management	Review new checkpoints, test updated LLMs, adjust prompt templates	Monthly / Quarterly
Retrieval QA	Measure recall@k, inspect failure cases, tune chunk sizes	Weekly
SQL Pipeline Validation	Verify schema changes, run safe-query tests on text-to-SQL logic	Weekly
System Monitoring	Track latency, token usage, GPU load, and error rates	Daily
Logs & Metrics Review	Aggregate Prometheus and Langfuse data, build trend dashboards	Continuous
Maintenance Tasks	Rotate secrets, clean Redis caches, backup data stores	Weekly
User Feedback Loop	Analyse chat logs, collect feedback, refine prompts and routing	Ongoing


⸻

Summary:
This team can sustain the system post-launch with predictable weekly and monthly routines. Automation of ingestion, monitoring, and retraining allows the project to scale efficiently without additional headcount until volumes or SLA commitments rise substantially.