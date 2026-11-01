# Manuscript-to-code map

| Component | Source entrypoints | Boundary |
|---|---|---|
| Primary vertex gradients | analysis/gradients/juzhen18.py → allnpy.py → Template.py → duiqi.py | Arrange raw outputs in Control/TMS × before/after folders before template/alignment |
| Parcellation | analysis/gradients/gradient_to_400.py; matrix_to_400.py | Additional atlas inputs required |
| Figure 1 | Study-design schematic | No numerical script is implied by the schematic |
| Figure 2a | analysis/figure2/a.py | Current 42-person, 19/23 grouping |
| Figure 2b | analysis/figure2/c.py | Spin enrichment; not obsolete main-2 b.py |
| Figure 2c | analysis/figure2/d.py | Graph metrics and spin associations; previous outputs required |
| Figure 3a | analysis/figure3/g.py | Clinical rows read from private CSV |
| Figure 3b–f | analysis/figure3/b.py, a.py, c.py, d.py, e.py | Order corresponds to panels b,c,d,e,f |
| Adjusted clinical and follow-up results | analysis/clinical/verify_clinical42.py; clinical_followup_audit.py | Primary ANCOVA42; complete follow-up28; 45-person audits are separate |
| Table 4 | analysis/table4/generate_table4.py | 41 paired-lesion patients |
| Figures 4–6 | analysis/figures456/run.py --figure 4 / 5 / 6 | Common tree; stages have distinct input requirements |
| Supplementary transcriptomics | analysis/transcriptomics/Gene.py; a.py–d.py | Optional branches are not evidence of analyses performed |

See individual notes for execution details. The obsolete main-2 b.py, older 18/24 ANCOVA implementations, raw data, intermediate arrays, cached results and duplicate work trees are excluded. Web-based enrichment requires its own gene-list and settings record.

