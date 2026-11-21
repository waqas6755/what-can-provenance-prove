-- Split-view check (Carlini et al., IEEE S&P 2024): a source shows the witness one
-- content and serves the pipeline another under the same name.  Receipts are either
-- content-addressed (bound to the item) or name-only.
module splitview

sig Name {}
sig Item { name : one Name }
one sig W {
  receipted : set Item,   -- what the authorised source signed a receipt for
  delivered : set Item,   -- what the pipeline actually received
  committed : set Item,
  used      : set Item
}
lone sig ContentReceipts {}   -- receipts bind content (hash); otherwise they bind names only
lone sig AdvSplit {}          -- source can serve different content than it receipted
lone sig Binding {}

fact {
  no AdvSplit implies W.delivered = W.receipted
  -- pipeline-side honesty here: capture and execution honest, to isolate the source channel
  W.committed = W.delivered
  W.used = W.delivered
}
pred Accepted {
  some Binding implies W.used = W.committed
  some ContentReceipts implies W.committed in W.receipted
  no ContentReceipts implies W.committed.name in W.receipted.name
}
pred G_auth_receipted { W.used in W.receipted }

check S1_content_receipts { (some AdvSplit and some Binding and some ContentReceipts and Accepted) implies G_auth_receipted } for 5
check S2_name_only_receipts { (some AdvSplit and some Binding and no ContentReceipts and Accepted) implies G_auth_receipted } for 5
