-- Targeted checks on v2 (assumption knobs and the v1 artefact).
open pipeline_v2_base

pred Full { some AdvSource and some AdvGen and some AdvTrans and some AdvCap and some AdvExec and no AdvWit and no AdvTEE }
pred AllMech { some BindFix and some WitSrc and some WitIn and some AttCap and some AttTrans and some Detector }

-- T1: perfect detector + binding + detector reaches G_zero under the full adversary
check T1_zero_with_perfect_detector { (Full and some Gen and some BindFix and some Detector and some PerfectDetector and Accepted) implies G_zero } for 5
-- T2: every mechanism, imperfect detector: G_zero still fails (residual = misses)
check T2_zero_all_mech_imperfect { (Full and some Gen and AllMech and no PerfectDetector and Accepted) implies G_zero } for 5
-- T3: every mechanism except the detector: no content claim at all (G_content reduces to G_zero)
check T3_content_without_detector { (Full and some Gen and some BindFix and some WitSrc and some WitIn and some AttCap and some AttTrans and no Detector and no PerfectDetector and Accepted) implies G_content } for 5
-- T4: poisoned authorised data passes every provenance check (Prop 5, negative half), shown as a satisfiable instance
run T4_authorised_poison_accepted { Full and some Gen and AllMech and no PerfectDetector and Accepted and some (W.used & W.bad & W.delivered) } for 3
-- T5: detector without binding: its verdict does not transfer to the model
check T5_detector_without_binding { (Full and no Gen and no BindFix and no BindGrind and some WitSrc and some WitIn and some AttCap and some AttTrans and some Detector and Accepted) implies G_content } for 5
-- T6: grinding: BindGrind + every other mechanism still leaves hidden influence through the seed
check T6_grinding { (Full and no Gen and some BindGrind and no BindFix and some WitSrc and some WitIn and some AttCap and some AttTrans and some Detector and Accepted) implies G_noinf } for 5
