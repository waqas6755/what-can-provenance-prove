"""Title/abstract decisions by the AI screener (Claude), 25 Oct 2026.
Single screener; NOT a human reviewer.  Inclusive rule: when unsure, pass to full text (U)."""
D = {}
def put(codes, ids):
    for i in ids.split(): D["R" + i.zfill(4)] = codes
put("I",  "1 2 5 7 15 19 20 21 24 25 26 28 29 30 33 34 35 36 38 46 47 52 53 75 76 81 93 96 108 117 118 120 121 125 147 149 151 155 159 165 178 187 188 194 196 205 206 212 219 223 232")
put("U",  "4 6 8 9 11 18 22 43 45 64 77 78 107 135 136 139 141 142 146 152 153 157 160 161 164 167 169 172 173 175 181 191 209 217 220")
put("E1", "17 23 40 42 51 54 58 60 62 63 79 80 82 84 85 86 87 88 90 94 102 105 106 111 114 124 126 127 131 137 140 143 144 163 180 185 186 193 195 201 208 210 213 226 230 231 234 243")
put("E1p","27 83 214")
put("E1m","244")
put("E2", "32 170 171 183 204")
put("E3", "37 61 91 168")
put("E5", "99 225 229")
# batch 2
put("I",  "255 258 262 271 275 288 296 297 305 310 319 323 327 329 342 343 345 354 369 370 396 417 435 444 448 450 453 455 469 470 478 484 497 500 503 512 514 515 516 522 541 547 551")
put("U",  "269 286 294 322 340 385 447 452 473 480 487 494 495 498 501 524 527 543")
put("E1", "245 246 251 256 261 264 265 302 303 306 307 308 309 315 316 318 321 326 335 337 338 339 346 348 349 350 351 355 372 380 382 383 388 404 408 412 428 429 436 449 454 456 467 468 482 488 502 504 510 295")
put("E1b","249 254 283 284 330 367 374 414 439 460 530")
put("E1p","289 377 402 406")
put("E1m","292 313 362 399 479")
put("E2", "250 276 368 407 477 505 519 531 532 536 538 545 546 548 550")
put("E3", "247 426 507")
put("E5", "458 525")
REASON = {"I": "include", "U": "include (unsure, to full text)", "E1": "E1 off-topic: ML as tool, or no provenance/integrity goal for ML artefacts",
          "E1p": "E1 privacy-only (no integrity/provenance component)", "E1m": "E1 membership inference framed as privacy attack, not data-use audit",
          "E1b": "E1 background mechanism, not ML-specific (kept on background list)", "E2": "E2 inference/output/ownership provenance (contrast list)",
          "E3": "E3 no technical mechanism or document type outside protocol", "E5": "E5 poisoning attack/defence with no provenance or binding component"}
