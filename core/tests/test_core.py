"""Run from core/: python -m unittest discover -s tests"""
import unittest

from graphrag_core import coverage, events, ranking
from graphrag_core.ppr import push_ppr, top
from graphrag_core.workflow import Workflow, sequence


def graph(edges):
    """Undirected weighted adjacency from (a, b, w) triples, as a neighbours() function that
    records which nodes were fetched."""
    adj = {}
    for a, b, w in edges:
        adj.setdefault(a, {})[b] = adj.get(a, {}).get(b, 0) + w
        adj.setdefault(b, {})[a] = adj.get(b, {}).get(a, 0) + w
    fetched = []

    def neighbours(nodes):
        fetched.extend(nodes)
        return {n: adj.get(n, {}) for n in nodes}
    return neighbours, fetched


class PushPPRTest(unittest.TestCase):
    def test_mass_concentrates_near_the_seed(self):
        nb, _ = graph([("a", "b", 1), ("b", "c", 1), ("c", "d", 1), ("d", "e", 1)])
        scores, _ = push_ppr({"a": 1.0}, nb)
        self.assertEqual(top(scores, 5), ["a", "b", "c", "d", "e"][:len(scores)])
        self.assertAlmostEqual(sum(scores.values()), 1.0, delta=0.01)

    def test_heavier_edges_carry_more(self):
        nb, _ = graph([("s", "heavy", 5), ("s", "light", 1)])
        scores, _ = push_ppr({"s": 1.0}, nb)
        self.assertGreater(scores["heavy"], scores["light"])

    def test_local_work_on_a_large_graph(self):
        # A 10 000-node path: the push only fetches the seed's neighbourhood
        nb, fetched = graph([(i, i + 1, 1) for i in range(10_000)])
        push_ppr({5000: 1.0}, nb, epsilon=1e-3)
        self.assertLess(len(fetched), 100)

    def test_isolated_seed_keeps_its_mass_and_empty_seeds_score_nothing(self):
        nb, _ = graph([])
        self.assertEqual(push_ppr({"x": 1.0}, nb)[0], {"x": 1.0})
        self.assertEqual(push_ppr({}, nb), ({}, 0))

    def test_fetch_cap(self):
        nb, fetched = graph([(0, i, 1) for i in range(1, 50)])
        push_ppr({0: 1.0}, nb, max_nodes=10)
        self.assertLessEqual(len(set(fetched)), 10)


class RankingTest(unittest.TestCase):
    def test_mixes_normalised_signals_and_drops_absent_ones(self):
        ranked = ranking.hybrid_rank(
            ["a", "b"], {"semantic": {"a": 0.8, "b": 0.4}, "ppr": {"a": 0.0, "b": 0.02}, "name": {}},
            {"semantic": 0.5, "ppr": 0.3, "name": 0.2})
        by_id = {r["id"]: r for r in ranked}
        self.assertAlmostEqual(by_id["a"]["score"], 0.5 / 0.8)
        self.assertAlmostEqual(by_id["b"]["score"], (0.5 * 0.5 + 0.3) / 0.8)
        self.assertEqual(ranked[0]["id"], "b")


class CoverageTest(unittest.TestCase):
    def test_uncovered_and_required_by(self):
        callers = {"f": ["g", "h"], "k": ["h"]}
        need = coverage.required_by(["f", "k"], lambda t: callers.get(t, []))
        self.assertEqual(need, {"g": ["f"], "h": ["f", "k"]})
        self.assertEqual(coverage.uncovered(need, ["g"]), ["h"])
        self.assertEqual(coverage.uncovered(["e1"], ["c1"], covers=lambda c, e: (c, e) == ("c1", "e1")), [])


class WorkflowTest(unittest.TestCase):
    def test_runs_steps_and_emits_agent_steps(self):
        seen = []
        wf = Workflow({"one": lambda s, e: {"n": 1, "_detail": "first"},
                       "two": lambda s, e: {"n": s["n"] + 1}}, "one", sequence("one", "two"))
        state = wf.run({}, seen.append)
        self.assertEqual(state, {"n": 2})
        self.assertEqual([e["agent"] for e in seen], ["one", "two"])
        self.assertTrue(seen[0]["detail"].startswith("first ("))
        self.assertEqual(seen[0]["type"], events.AGENT_STEP)

    def test_loops_are_bounded(self):
        wf = Workflow({"a": lambda s, e: {}}, "a", lambda cur, s: "a", max_steps=3)
        with self.assertRaises(RuntimeError):
            wf.run({}, lambda e: None)


if __name__ == "__main__":
    unittest.main()
