# Figure source map

Paths below are relative to the release root. Each figure is supplied as PNG and editable SVG in figures/.

| Figure | Result tables | Code |
|---|---|---|
| Figure_1_age | bi13_scores.json; age_selected_reactions.json; donors.json | code/analyze.py; code/figures.py |
| Figure_2_associations | all21_scores.json; correlations.json; donors.json | code/analyze.py; code/figures.py |
| Figure_S1_high_low | high_low.json | code/analyze.py; code/figures.py |
| Figure_S2_network_CoQ | age_all_reactions.json; age_all_ubiquinone_reactions.json | code/analyze.py; code/figures.py |
| Figure_S3_sensitivity | correlations.json; partial_correlations.json; age_interactions.json | code/analyze.py; code/figures.py |
| Figure_global_reaction_ranks | global_reaction_ranking.json; global_focal_reaction_ranking.json | code/global_ranking.py |
| Figure_global_pathways_1 / 2 | global_pathway_ranking.json | code/global_ranking.py |
| Figure_global_pathway_heatmap | global_pathway_donor_indices.json | code/global_ranking.py |
| depth_control | depth_control/all_results.json; donor_changes.json; summary.json | code/depth_control.py |

Result tables are under tables/. Frozen model input and reaction penalties are under data/. Raw preparation and solver execution are separate from deposited-score reproduction. Historical exploration records remain in tables/historical_grid_*; original depth-control execution scripts are retained in code/depth_control_source/ as provenance.
