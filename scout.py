Hi Garvit, team,

Thanks for sharing the observations. This is a critical part of the pipeline, so I want to make sure we approach it in a structured and aligned way.

Given the complexity and variability in the PPT layouts, I suggest we proceed as follows:
	1.	PPT Variant Coverage
@Gupta, Himanshi and @Satish chalak, Rohit — please test a broad range of case study PPT variants using both existing PPTX-based logic and Docling, so we can understand concrete strengths, gaps, and failure modes. I’ll share the scripts and evaluation checklist to standardise this.
	2.	Granularity in PPT Extraction
Based on initial observations, PPTs should not be treated as a single unstructured blob. Wherever possible, we should leverage native PPT layers and structure to preserve granularity before considering heavier processing.
	3.	Knowledge Graph Approach
On KG, I’m not convinced yet that it adds sufficient value for the current scope. Let’s keep this on hold until we have clear evidence that it materially improves outcomes.
	4.	Layout-Based Options
@Nagpal, Garvit — it would be helpful if you could look into layout-aware extraction approaches that are practical within the BofA environment. We can evaluate these once we see results from baseline testing.
	5.	Parallel Workstreams
@Ashok, Gadde and @Pandit, Shivansh — please continue with the KG-related exploration as currently guided by @Shivaraju, Madhura. We’ll reassess once the PPT findings are consolidated.
	6.	Integration & Ownership
I’ll continue driving the overall ingestion design and integration, and once we have empirical results across PPT variants, I’ll finalise the gating logic and tooling decisions.

This should give us clarity based on evidence rather than assumptions, while keeping efforts coordinated.

Thanks everyone — let’s sync once the first round of findings is in.

Best,
Abel
