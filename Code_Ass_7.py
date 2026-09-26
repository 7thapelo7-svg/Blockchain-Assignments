
from __future__ import annotations

import hashlib
import heapq
import json
import math
import multiprocessing as mp
import queue
import random
import sys
import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable



# blocks, nodes, validation, fork choice, confirmations

Block = dict[str, Any]
Chain = list[Block]

DIFFICULTY = 2          # leading hex zeros (toy classroom setting)
GENESIS_PREV = "0" * 64



# --------------------Hashing and mining

def _sha(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def header_hash(index, timestamp, txs, previous_hash, nonce) -> str:
    """SHA-256 over canonical JSON (sort_keys=True) of the block header."""
    payload = json.dumps(
        {"index": index, "timestamp": timestamp, "transactions": txs,
         "previous_hash": previous_hash, "nonce": nonce},
        sort_keys=True,
    )
    return _sha(payload)


def mine_block(index: int, txs: list, previous_hash: str,
               difficulty: int = DIFFICULTY, timestamp: float | None = None) -> Block:
    """Proof-of-work: search nonces until the hash has `difficulty` leading zeros."""
    ts = time.time() if timestamp is None else timestamp
    nonce = 0
    target = "0" * difficulty
    while True:
        h = header_hash(index, ts, txs, previous_hash, nonce)
        if h.startswith(target):
            return {"index": index, "timestamp": ts, "transactions": txs,
                    "previous_hash": previous_hash, "nonce": nonce,
                    "hash": h, "difficulty": difficulty}
        nonce += 1


def make_genesis(difficulty: int = DIFFICULTY) -> Block:
    # Fixed timestamp so that every node derives the *same* genesis block.
    return mine_block(0, [{"note": "genesis"}], GENESIS_PREV, difficulty, timestamp=0.0)



# Nodes

@dataclass
class Node:
    """A simulated peer holding its own (possibly divergent) view of the chain."""
    name: str
    chain: Chain = field(default_factory=list)

    def tip(self) -> Block:
        if not self.chain:
            raise ValueError(f"node {self.name} has an empty chain")
        return self.chain[-1]

    def height(self) -> int:
        return len(self.chain) - 1 if self.chain else -1

    def mine(self, txs: list, difficulty: int = DIFFICULTY,
             timestamp: float | None = None) -> Block:
        """Mine a new block on this node's own tip and append it locally."""
        blk = mine_block(self.height() + 1, txs, self.tip()["hash"], difficulty, timestamp)
        self.chain.append(blk)
        return blk

    def receive(self, candidate: Chain) -> bool:
        """Apply fork choice to (own chain, candidate). Returns True on a reorg."""
        old_tip = self.tip()["hash"] if self.chain else None
        self.chain = resolve_fork([self.chain, candidate])
        return self.tip()["hash"] != old_tip

    def summary(self) -> str:
        return (f"{self.name}: height={self.height()} tip={self.tip()['hash'][:10]}... "
                f"valid={verify_chain(self.chain)} work={chain_work(self.chain)}")



# ----------------------------Validation and fork choice

def verify_chain(chain: Chain, difficulty: int = DIFFICULTY) -> bool:
    """True iff every hash recomputes, meets PoW, links to its parent and is indexed."""
    if not chain:
        return False
    for i, block in enumerate(chain):
        expected = header_hash(block["index"], block["timestamp"], block["transactions"],
                               block["previous_hash"], block["nonce"])
        if block.get("hash") != expected:
            return False
        diff = int(block.get("difficulty", difficulty))
        if not str(block["hash"]).startswith("0" * diff):
            return False
        if i == 0:
            continue
        if block["previous_hash"] != chain[i - 1]["hash"]:
            return False
        if block["index"] != i:
            return False
    return True


def chain_length(chain: Chain) -> int:
    """Longest-chain score."""
    return len(chain)


def chain_work(chain: Chain) -> int:
    """Heaviest-work score: sum of per-block difficulty."""
    return sum(int(b.get("difficulty", DIFFICULTY)) for b in chain)


def tie_break(chains: list[Chain]) -> Chain:
    """Documented tie policy: the lexicographically smaller tip hash wins."""
    return min(chains, key=lambda c: c[-1]["hash"])


def resolve_fork(chains: list[Chain],
                 verifier: Callable[[Chain], bool] | None = None,
                 score: Callable[[Chain], int] = chain_work) -> Chain:
    """Filter to valid chains, maximise `score`, break ties, return a deep copy."""
    verify = verifier or verify_chain
    valid = [c for c in chains if verify(c)]
    if not valid:
        raise ValueError("no valid chain")
    best = max(score(c) for c in valid)
    tied = [c for c in valid if score(c) == best]
    chosen = tied[0] if len(tied) == 1 else tie_break(tied)
    return deepcopy(chosen)


def sync_nodes(nodes: list[Node]) -> Chain:
    """Resolve all node chains and set every node to the winner."""
    winner = resolve_fork([n.chain for n in nodes])
    for node in nodes:
        node.chain = deepcopy(winner)
    return winner


def fork_point(a: Chain, b: Chain) -> int:
    """Height of the last common ancestor of two chains (-1 if none)."""
    h = -1
    for x, y in zip(a, b):
        if x["hash"] != y["hash"]:
            break
        h = x["index"]
    return h


def orphaned_blocks(losing: Chain, canonical: Chain) -> list[Block]:
    """Valid blocks on `losing` that are not on the canonical chain."""
    keep = {b["hash"] for b in canonical}
    return [b for b in losing if b["hash"] not in keep]



# ------------------------------------------------Confirmation depth

def find_tx_height(chain: Chain, tx_id: str) -> int | None:
    for block in chain:
        for tx in block.get("transactions", []):
            if tx.get("id") == tx_id:
                return int(block["index"])
    return None


def confirmations(chain: Chain, tx_id: str) -> int:
    """k = H - h + 1, or 0 if the transaction is not on this chain."""
    h = find_tx_height(chain, tx_id)
    if h is None or not chain:
        return 0
    return int(chain[-1]["index"]) - h + 1


def settled_for_policy(chain: Chain, tx_id: str, required_k: int) -> bool:
    return confirmations(chain, tx_id) >= required_k


def attacker_success_probability(q: float, k: int) -> float:
    """Nakamoto (2008, s.11): probability an attacker with hash share q ever
    catches up from k blocks behind, using the Poisson approximation."""
    p = 1.0 - q
    if q >= p:
        return 1.0
    lam = k * (q / p)
    s = 1.0
    for i in range(k + 1):
        poisson = math.exp(-lam) * lam ** i / math.factorial(i)
        s -= poisson * (1 - (q / p) ** (k - i))
    return s



#------------------------- End-to-end simulater

def demo_fork_scenario() -> dict[str, Any]:
    g = make_genesis()
    a, b, c = (Node(n, [deepcopy(g)]) for n in "ABC")
    a.mine([{"id": "pay-thapelo", "sender": "Abigail", "amount": 10000, "currency": "ZAR"}])
    b.mine([{"id": "pay-kgothalang", "sender": "Abigail", "amount": 10000, "currency": "ZAR"}])
    a.mine([{"note": "extend-A"}])
    before = {"A": a.height(), "B": b.height(), "C": c.height()}
    sync_nodes([a, b, c])
    return {"heights_before": before, "tip_after": a.tip()["hash"],
            "canonical_tx": a.chain[1]["transactions"][0]["id"],
            "confirmations_thapelo": confirmations(a.chain, "pay-thapelo"),
            "confirmations_kgothalang": confirmations(a.chain, "pay-kgothalang")}


#----------------------------Scenario runs

G = make_genesis()
PAY_THAPELO = [{"id": "pay-thapelo", "sender": "Abigail", "recipient": "Thapelo",
              "amount": 10000, "currency": "ZAR"}]
PAY_KGOTHALANG = [{"id": "pay-kgothalang", "sender": "Abigail", "recipient": "Kgothalang",
            "amount": 10000, "currency": "ZAR"}]


T0 = 1_789_000_000          
OFFSET = {"A": 0, "B": 7, "C": 13}


def ts(node: Node) -> float:
    """Deterministic timestamp: ~10-minute block spacing plus a per-node offset."""
    return T0 + 600 * (node.height() + 1) + OFFSET[node.name[0]]


def fresh_nodes():
    return [Node(n, [deepcopy(G)]) for n in "ABC"]


def forked():
    A, B, C = fresh_nodes()
    A.mine(PAY_THAPELO, timestamp=ts(A))
    B.mine(PAY_KGOTHALANG, timestamp=ts(B))
    return A, B, C


def s1_nodes():
    print("== 1. Three nodes share one genesis ==")
    A, B, C = fresh_nodes()
    for n in (A, B, C):
        print(" ", n.summary())
    print("  Shared tip?", A.tip()["hash"] == B.tip()["hash"] == C.tip()["hash"])


def s2_fork():
    print("== 2. Create a fork: A and B mine competing block-1s ==")
    A, B, C = forked()
    for n in (A, B, C):
        print(" ", n.summary())
    print("  A and B tips differ?        ", A.tip()["hash"] != B.tip()["hash"])
    print("  Both chains valid?          ", verify_chain(A.chain), verify_chain(B.chain))
    print("  Same parent (genesis)?      ",
          A.chain[1]["previous_hash"] == B.chain[1]["previous_hash"] == G["hash"])
    print("  Fork point (last common h): ", fork_point(A.chain, B.chain))
    print("  A block 1 tx:", A.chain[1]["transactions"][0]["id"],
          "| B block 1 tx:", B.chain[1]["transactions"][0]["id"])


def s3_resolve():
    print("== 3. Resolve: extend A's branch, then sync all nodes ==")
    A, B, C = forked()
    A.mine([{"note": "extend-A"}], timestamp=ts(A))
    print("  Before sync:")
    for n in (A, B, C):
        print("   ", n.summary())
    losing_B = deepcopy(B.chain)
    canonical = sync_nodes([A, B, C])
    print("  After sync:")
    for n in (A, B, C):
        print("   ", n.summary())
    print("  All tips equal?", A.tip()["hash"] == B.tip()["hash"] == C.tip()["hash"])
    print("  Canonical block-1 tx:", canonical[1]["transactions"][0]["id"])
    orphans = orphaned_blocks(losing_B, canonical)
    print("  Orphaned (stale) blocks:", [(b["index"], b["hash"][:10]) for b in orphans])
    print("  Orphan still valid alone?", verify_chain([deepcopy(G)] + orphans))
    print("  confirmations(pay-kgothalang) on canonical =", confirmations(canonical, "pay-kgothalang"))


def s4_invalid_and_rules():
    print("== 4. Validate before compare; longest vs heaviest ==")
    A, B, _ = forked()
    bad = deepcopy(A.chain)
    bad[1] = dict(bad[1]); bad[1]["transactions"] = [{"id": "pay-thapelo", "amount": 99999}]
    for i in range(2, 6):                       # make it *longer* on top of the forgery
        bad.append(mine_block(i, [{"note": f"junk-{i}"}], bad[-1]["hash"], timestamp=T0 + 600 * i))
    print(f"  Tampered chain: length={len(bad)} valid={verify_chain(bad)}")
    w = resolve_fork([bad, B.chain])
    print(f"  Winner: length={len(w)} tx={w[1]['transactions'][0]['id']} (tampered chain rejected)")
    easy = [deepcopy(G)]
    for i in range(1, 5):
        easy.append(mine_block(i, [{"note": f"easy-{i}"}], easy[-1]["hash"], difficulty=1,
                                timestamp=T0 + 600 * i))
    hard = [deepcopy(G)]
    for i in range(1, 3):
        hard.append(mine_block(i, [{"note": f"hard-{i}"}], hard[-1]["hash"], difficulty=4,
                                timestamp=T0 + 600 * i))
    for name, c in (("long-easy", easy), ("short-hard", hard)):
        print(f"  {name:10s}: length={chain_length(c)} work={chain_work(c)} valid={verify_chain(c)}")
    lw = resolve_fork([easy, hard], score=chain_length)
    hw = resolve_fork([easy, hard], score=chain_work)
    print("  Longest-chain rule picks :", "long-easy" if lw[-1]["hash"] == easy[-1]["hash"] else "short-hard")
    print("  Heaviest-work rule picks :", "long-easy" if hw[-1]["hash"] == easy[-1]["hash"] else "short-hard")


def s5_tie():
    print("== 5. Edge case: a tie at equal work ==")
    A, B, C = forked()
    print(f"  work(A)={chain_work(A.chain)} work(B)={chain_work(B.chain)} -> tie")
    print("  A tip:", A.tip()["hash"][:16], "| B tip:", B.tip()["hash"][:16])
    w1 = resolve_fork([A.chain, B.chain]); w2 = resolve_fork([B.chain, A.chain])
    print("  Winner (A,B order):", w1[-1]["hash"][:16], w1[1]["transactions"][0]["id"])
    print("  Winner (B,A order):", w2[-1]["hash"][:16], w2[1]["transactions"][0]["id"])
    print("  Order-independent? ", w1[-1]["hash"] == w2[-1]["hash"])
    # First-seen alternative: C's view depends on arrival order
    c1 = Node("C1", [deepcopy(G)]); c2 = Node("C2", [deepcopy(G)])
    c1.chain = deepcopy(A.chain); c2.chain = deepcopy(B.chain)   # first-seen
    print("  First-seen policy: C1 keeps", c1.chain[1]["transactions"][0]["id"],
          "| C2 keeps", c2.chain[1]["transactions"][0]["id"], "-> views stay split")
    # The tie is only truly broken by the next block
    B.mine([{"note": "extend-B"}], timestamp=ts(B))
    w3 = resolve_fork([A.chain, B.chain])
    print("  B mines block 2 -> winner:", w3[1]["transactions"][0]["id"],
          f"(work {chain_work(w3)} vs {chain_work(A.chain)}) - tie-break no longer used")


def s6_delay():
    print("== 6. Edge case: delayed block arrival and reorg at node C ==")
    A, B, C = forked()
    A.mine([{"note": "extend-A"}], timestamp=ts(A))
    chain_A, chain_B = deepcopy(A.chain), deepcopy(B.chain)
    print("  t0", C.summary())
    reorg = C.receive(chain_B)
    print("  t1 C hears B first  ->", C.summary(), "| reorg:", reorg)
    print("       C sees pay-kgothalang with k =", confirmations(C.chain, "pay-kgothalang"))
    reorg = C.receive(chain_A)
    print("  t2 A's heavier chain arrives ->", C.summary(), "| reorg:", reorg)
    print("       k(pay-kgothalang) =", confirmations(C.chain, "pay-kgothalang"),
          "| k(pay-thapelo) =", confirmations(C.chain, "pay-thapelo"))
    reorg = C.receive(chain_B)                   # stale B block re-delivered late
    print("  t3 stale B chain re-delivered ->", "reorg:", reorg, "(ignored, lower work)")


def s7_confirm():
    print("== 7. Confirmation depth k = H - h + 1 for the R10 000 payment ==")
    A, B, C = forked()
    for i in range(2, 7):
        A.mine([{"note": f"block-{i}"}], timestamp=ts(A))
    for H in range(1, 7):
        k = confirmations(A.chain[:H + 1], "pay-thapelo")
        print(f"  tip H={H}: k={k}  coffee(k>=1)={settled_for_policy(A.chain[:H+1],'pay-thapelo',1)!s:5}"
              f" policy(k>=3)={settled_for_policy(A.chain[:H+1],'pay-thapelo',3)!s:5}"
              f" OTC(k>=6)={settled_for_policy(A.chain[:H+1],'pay-thapelo',6)}")
    print("  Attacker catch-up probability P(q, k)  [Nakamoto 2008, s.11]")
    print("    k  " + "  ".join(f"q={q:<5}" for q in (0.10, 0.20, 0.30)))
    for k in (0, 1, 2, 3, 6):
        print(f"    {k}  " + "  ".join(f"{attacker_success_probability(q, k):<7.4f}" for q in (0.10, 0.20, 0.30)))


#  nodes A, B, C as separate processes with network latency

DIFF = 4                      # ~0.3-1 s per block per node on a laptop
LATENCY = {("A", "B"): 0.05, ("B", "A"): 0.05,
           ("A", "C"): 0.60, ("C", "A"): 0.60,
           ("B", "C"): 0.60, ("C", "B"): 0.60}
BATCH = 500


def node_proc(name, inbox, outbox, results, mine_until, stop_at, seed):
    rng = random.Random(seed)
    chain = [make_genesis()]
    seen_blocks = {chain[0]["hash"]: chain[0]}
    reorgs, mined = 0, 0
    nonce, ts, txs = 0, time.time(), None
    while time.time() < stop_at:
        # 1) drain inbox and apply fork choice
        try:
            while True:
                cand = inbox.get_nowait()
                for b in cand:
                    seen_blocks[b["hash"]] = b
                old = chain[-1]["hash"]
                new = resolve_fork([chain, cand])
                if new[-1]["hash"] != old:
                    # a reorg = our old tip is no longer an ancestor of the new tip
                    if old not in {b["hash"] for b in new}:
                        reorgs += 1
                    chain = new
                    txs = None          # restart mining on the new tip
        except queue.Empty:
            pass
        # 2) mine a batch of nonces (only during the mining window)
        if time.time() >= mine_until:
            time.sleep(0.01)
            continue
        if txs is None:
            txs = [{"id": f"{name}-{len(chain)}-{rng.randint(0, 9999)}", "miner": name}]
            nonce, ts = 0, time.time()
        idx, prev = len(chain), chain[-1]["hash"]
        for _ in range(BATCH):
            h = header_hash(idx, ts, txs, prev, nonce)
            if h.startswith("0" * DIFF):
                blk = {"index": idx, "timestamp": ts, "transactions": txs,
                       "previous_hash": prev, "nonce": nonce, "hash": h, "difficulty": DIFF}
                chain = chain + [blk]
                seen_blocks[h] = blk
                mined += 1
                outbox.put((name, chain))
                txs = None
                break
            nonce += 1
    canonical_hashes = {b["hash"] for b in chain}
    stale = [h for h in seen_blocks if h not in canonical_hashes]
    results.put({"name": name, "chain": chain, "reorgs": reorgs, "mined": mined,
                 "stale_seen": len(stale)})


def run_network(duration: float = 6.0, seed: int = 7):
    names = ["A", "B", "C"]
    inboxes = {n: mp.Queue() for n in names}
    outbox, results = mp.Queue(), mp.Queue()
    start = time.time()
    mine_until, stop_at = start + duration, start + duration + 2.0   # 2 s quiet period
    procs = [mp.Process(target=node_proc,
                        args=(n, inboxes[n], outbox, results, mine_until, stop_at, seed + i))
             for i, n in enumerate(names)]
    for p in procs:
        p.start()
    pending: list = []            # (deliver_at, seq, dest, chain)
    seq, sent = 0, 0
    while time.time() < stop_at:
        try:
            src, chain = outbox.get(timeout=0.01)
            for dst in names:
                if dst != src:
                    heapq.heappush(pending, (time.time() + LATENCY[(src, dst)], seq, dst, chain))
                    seq += 1
        except queue.Empty:
            pass
        while pending and pending[0][0] <= time.time():
            _, _, dst, chain = heapq.heappop(pending)
            inboxes[dst].put(chain)
            sent += 1
    out = {r["name"]: r for r in (results.get() for _ in names)}
    for p in procs:
        p.join()

    print(f"Separate-process simulation: {duration:.0f}s mining + 2s quiet, difficulty={DIFF}")
    print("Link latency: A<->B 0.05s, A<->C 0.60s, B<->C 0.60s | messages delivered:", sent)
    for n in names:
        r = out[n]
        print(f"  {n}: mined={r['mined']:2d} height={len(r['chain'])-1:2d} "
              f"tip={r['chain'][-1]['hash'][:10]}... reorgs={r['reorgs']:2d} "
              f"stale_blocks_seen={r['stale_seen']:2d} valid={verify_chain(r['chain'], DIFF)}")
    tips = {out[n]["chain"][-1]["hash"] for n in names}
    print("  Converged on one tip after quiet period?", len(tips) == 1)
    canon = out["A"]["chain"]
    by_miner = {n: sum(1 for b in canon[1:] if b["transactions"][0].get("miner") == n) for n in names}
    print("  Canonical blocks by miner:", by_miner,
          "| total mined:", sum(out[n]["mined"] for n in names),
          "| orphaned:", sum(out[n]["mined"] for n in names) - (len(canon) - 1))


#-------------------------------- Self-tests

def branch(tag, n=1, base=None):
    c = deepcopy(base) if base else [deepcopy(G)]
    for _ in range(n):
        c.append(mine_block(len(c), [{"id": f"{tag}-{len(c)}"}], c[-1]["hash"]))
    return c


def test_genesis_is_shared_and_valid():
    assert make_genesis()["hash"] == G["hash"]
    assert verify_chain([G])

def test_node_api():
    n = Node("A")
    assert n.height() == -1
    n.chain.append(deepcopy(G))
    assert n.height() == 0 and n.tip()["index"] == 0

def test_fork_both_valid_same_parent():
    a, b = branch("a"), branch("b")
    assert verify_chain(a) and verify_chain(b)
    assert a[1]["previous_hash"] == b[1]["previous_hash"] == G["hash"]
    assert fork_point(a, b) == 0

def test_longer_valid_chain_wins_and_sync_converges():
    a, b = branch("a", 2), branch("b", 1)
    nodes = [Node("A", a), Node("B", b), Node("C", [deepcopy(G)])]
    w = sync_nodes(nodes)
    assert len(w) == 3 and w[1]["transactions"][0]["id"] == "a-1"
    assert len({n.tip()["hash"] for n in nodes}) == 1
    assert orphaned_blocks(b, w) == [b[1]]

def test_invalid_longer_chain_never_wins():
    bad = branch("x", 4)
    bad[2] = dict(bad[2]); bad[2]["transactions"] = [{"id": "forged"}]
    good = branch("g", 1)
    assert not verify_chain(bad)
    assert resolve_fork([bad, good])[-1]["hash"] == good[-1]["hash"]

def test_tie_break_is_order_independent():
    a, b = branch("a"), branch("b")
    assert resolve_fork([a, b])[-1]["hash"] == resolve_fork([b, a])[-1]["hash"]
    assert resolve_fork([a, b])[-1]["hash"] == min(a[-1]["hash"], b[-1]["hash"])

def test_heaviest_differs_from_longest_when_difficulty_varies():
    easy = [deepcopy(G)]
    for i in range(1, 5):
        easy.append(mine_block(i, [{"id": i}], easy[-1]["hash"], difficulty=1))
    hard = [deepcopy(G)]
    for i in range(1, 3):
        hard.append(mine_block(i, [{"id": i}], hard[-1]["hash"], difficulty=3))
    assert chain_length(easy) > chain_length(hard) and chain_work(hard) > chain_work(easy)
    assert resolve_fork([easy, hard], score=chain_length)[-1]["hash"] == easy[-1]["hash"]
    assert resolve_fork([easy, hard])[-1]["hash"] == hard[-1]["hash"]

def test_delayed_arrival_reorg():
    c = Node("C", [deepcopy(G)])
    b, a = branch("b", 1), branch("a", 2)
    assert c.receive(b) and confirmations(c.chain, "b-1") == 1
    assert c.receive(a) and confirmations(c.chain, "b-1") == 0
    assert not c.receive(b)          # stale re-delivery is ignored

def test_confirmation_depth():
    c = branch("p", 1)
    assert confirmations(c, "p-1") == 1
    c = branch("pad", 4, base=c)
    assert confirmations(c, "p-1") == 5
    assert confirmations(c, "missing") == 0

def test_attacker_probability_matches_nakamoto():
    assert abs(attacker_success_probability(0.1, 5) - 0.0009137) < 1e-6
    assert abs(attacker_success_probability(0.3, 5) - 0.1773523) < 1e-6

def test_demo_scenario():
    s = demo_fork_scenario()
    assert s["canonical_tx"] == "pay-thapelo" and s["confirmations_kgothalang"] == 0

def run_tests() -> None:
    tests = [test_genesis_is_shared_and_valid, test_node_api, test_fork_both_valid_same_parent, test_longer_valid_chain_wins_and_sync_converges, test_invalid_longer_chain_never_wins, test_tie_break_is_order_independent, test_heaviest_differs_from_longest_when_difficulty_varies, test_delayed_arrival_reorg, test_confirmation_depth, test_attacker_probability_matches_nakamoto, test_demo_scenario]
    for fn in tests:
        fn()
        print(f"  PASS  {fn.__name__}")
    print(f"All {len(tests)} self-tests passed.")


SECTIONS = {"1": s1_nodes, "2": s2_fork, "3": s3_resolve, "4": s4_invalid_and_rules,
            "5": s5_tie, "6": s6_delay, "7": s7_confirm}


def run_demo(which: str = "all") -> None:
    for key, fn in SECTIONS.items():
        if which in ("all", key):
            fn()
            print()


if __name__ == "__main__":
    args = sys.argv[1:]
    mode = args[0] if args else "everything"
    if mode == "demo":
        run_demo(args[1] if len(args) > 1 else "all")
    elif mode == "network":
        run_network(float(args[1]) if len(args) > 1 else 6.0,
                    int(args[2]) if len(args) > 2 else 7)
    elif mode == "test":
        run_tests()
    else:
        run_demo("all")
        print("== Self-tests ==")
        run_tests()
        print()
        run_network(10.0, 7)
