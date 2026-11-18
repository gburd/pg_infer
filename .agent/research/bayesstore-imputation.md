# BayesStore and the imputation question

Research report, 2026-11-18.  Written from the primary papers, not
secondary summaries.  Every number below is quoted from a paper I read;
where I could not verify something I say so.

## 0. Verdict up front

The system the user remembers is **BayesStore**, UC Berkeley, VLDB 2008.
The identification is solid: Berkeley, Bayesian networks held as
first-class database objects, and missing attribute values as the unit of
uncertainty.  That is the user's description almost word for word.

Two nearby Berkeley systems are worth knowing because they are often
conflated with it, and one of them matches "fill in missing data" even more
literally than BayesStore does:

* **BBQ** (VLDB 2004) is Berkeley, and its entire purpose is answering
  queries about sensor values the system never read, from a learned joint
  distribution.  But its model is a time-varying multivariate Gaussian,
  not a Bayesian network, and the point of the work is deciding which
  sensors to *not* read.
* **MauveDB** (SIGMOD 2006) is **Maryland and MIT, not Berkeley** - Amol
  Deshpande had moved to UMD and Samuel Madden was at MIT.  The user's
  instinct here was right.

Ruled out by institution: Tuffy and DeepDive (Wisconsin, then Stanford),
Lahar (Washington), Trio (Stanford), MystiQ (Washington), MCDB (Florida and
IBM Almaden), Orion (Purdue), PrDB (Maryland).  Citations in §7.

The answer to the question you actually care about is in §8.  Short form:
the analogy is superficial where it matters and sound in one narrow place
that is not the place people usually mean.

## 1. Citations

**The paper.**  Daisy Zhe Wang, Eirinaios Michelakis, Minos Garofalakis,
and Joseph M. Hellerstein.  "BayesStore: Managing Large, Uncertain Data
Repositories with Probabilistic Graphical Models."  PVLDB 1(1):340-351,
VLDB 2008, Auckland.  Affiliations on the paper: Wang, Michelakis, and
Hellerstein at UC Berkeley EECS; Garofalakis jointly Yahoo! Research and
Berkeley.  DOI 10.14778/1453856.1453896, PDF at
<https://www.vldb.org/pvldb/vol1/1453896.pdf>.  171 citations per OpenAlex.

**The fullest primary source, and the one to read.**  Daisy Zhe Wang's
Berkeley dissertation, "Extracting and Querying Probabilistic Information
in BayesStore," UCB/EECS-2011-131, 13 December 2011, 157 pages.  Committee:
Michael J. Franklin (chair), Minos Garofalakis, Joseph M. Hellerstein, Cari
Kaufman.  <https://www2.eecs.berkeley.edu/Pubs/TechRpts/2011/EECS-2011-131.html>
(PDF under .../2011/Archive/EECS-2011-131.pdf; the eScholarship mirror at
escholarship.org/uc/item/3557w390 is behind a CloudFront block that returns
403 to scripted fetches).  It contains everything in the VLDB paper plus
the in-database inference implementations, and it is where the design is
actually pinned down.

**The rest of the line, in order:**

* Michelakis, Wang, Garofalakis, Hellerstein, "Granularity Conscious
  Modeling for Probabilistic Databases," DUNE/ICDMW 2007,
  doi 10.1109/icdmw.2007.63.  The precursor: hierarchical FOBNs and the
  learning problem.
* Wang, Michelakis, Franklin, Garofalakis, Hellerstein, "Probabilistic
  Declarative Information Extraction," ICDE 2010,
  doi 10.1109/icde.2010.5447844.
* Wang, Franklin, Garofalakis, Hellerstein, "Querying Probabilistic
  Information Extraction," PVLDB 3(1), 2010, doi 10.14778/1920841.1920974.
* Wang, Franklin, Garofalakis, Hellerstein, Michael Wick, "Hybrid
  In-Database Inference for Declarative Information Extraction," SIGMOD
  2011, doi 10.1145/1989323.1989378.  (Garofalakis by then at Technical
  University of Crete.)

Note that the VLDB 2008 paper defers four things to "the full version of
this paper" - the FOBN acyclicity constraints, the Bayes Ball details, the
join special case, and the SQL extension syntax.  **I could not find a
public copy of that full version.**  The dissertation covers three of the
four, but *not* the SQL `INFER` syntax, which appears only in the VLDB
paper's §4.5.  So on the exact surface syntax the published record is one
example long.

## 2. The data model

A BayesStore probabilistic database is a pair

```
DB^p = <R, F>
```

`R` is a collection of **incomplete relations**; `F` is a probability
distribution held *separately* from them.  This separation is the design
decision the whole system turns on, and the paper is explicit that it is a
departure from everything before it: "Rather than attaching probabilities
to uncertain database elements, BayesStore maintains the probability
distribution F over the incomplete relations as a separate entity."
Trio, MystiQ, MayBMS, and Orion all attach probabilities to tuples or
values.  BayesStore does not.

**Incomplete relation.**  Schema splits into deterministic attributes `A^d`
(which must include every candidate and foreign key) and probabilistic
attributes `A^p`.  A deterministic attribute always holds a legal value or
an ordinary SQL NULL.  A probabilistic attribute may be **present or
missing**.  Present means *evidence*.  Missing means a random variable
`X_j` ranging over `dom(A_i)`.  The running example:

```
Sensor1(Time, Room, Sid, temp^p, light^p)
Sensor2(Time, Room, humidity^p, light^p)
```

with blanks in the `temp^p` and `light^p` columns.  Six tuples, sixteen
probabilistic cells, so sixteen RVs.

**Possible world.**  Exactly this: every complete assignment of values to
all the `X_i` is one possible world.  Nothing more exotic.  Note that
tuple-*existence* uncertainty is not primitive - a relation that needs it
carries a probabilistic boolean attribute `Exist^p`, and §4 shows that
query processing introduces that attribute whether the user asked for it or
not.

**F, factored.**  Stored naively, F is linear in the number of possible
worlds (2^16 for the toy example).  Two compressions:

1. An ordinary **Bayesian network**, one RV per probabilistic attribute
   value per tuple, local CPT `Pr[X_i | Pa(X_i)]` per node, joint =
   product of the locals.  The 16-dimensional table becomes 12
   two-dimensional ones.  The paper distinguishes **horizontal**
   correlations (within a tuple) from **vertical** ones (across tuples)
   and that distinction drives the query algorithms.
2. A **First-Order Bayesian Network (FOBN)**, because the propositional BN
   "requires one RV per probabilistic attribute, per tuple, per relation."

**FOBN is the paper's actual contribution.**  Three definitions:

* **Entity set** `R.E` - a sub-population of tuples, specified
  *declaratively as a relational algebra expression*:
  `R.E = π_K'(σ_cond(R))` over the deterministic attributes.  The maximal
  entity set projects on a key, giving one entity per tuple.  Coarser ones
  give one entity per group, and all tuples in an entity's extent take the
  same value in every possible world.
* **Stripe** `S<R.A^p, E>` - a first-order RV: the population of RVs for
  one probabilistic attribute across one entity set.
* **First-order factor** `φ({S_i}, {f_i}, CPT)` - a child stripe, parent
  stripes, surjective **mappings** `f[S_p, S_c]: E_c -> E_p` written as
  first-order formulas over the entity-set schemas, and **one CPT shared
  across the whole population**.

So a correlation between light and temperature that holds for every sensor
in the building is *one* factor with *one* CPT, not one per tuple.  Cross-
relation factors are allowed along key/foreign-key paths, with the
constraint that the parent's foreign key be part of the child's primary key
and that both entity sets carry the same selection predicate.  A child RV
may have at most one parent per factor, since a single first-order edge
cannot express a multi-variable correlation.

PRMs (Friedman, Getoor, Koller, Pfeffer, IJCAI 1999) are a strict special
case: a PRM's first-order RVs are tied to schema attributes and tuples are
assumed independent, where a BayesStore stripe can be any declaratively
carved sub-population and vertical correlations are allowed.

**Where the CPTs physically live.**  The VLDB paper says only that models
are stored in separate relational tables.  The dissertation gives a
concrete schema for the CRF instance:

```
MR(token, label, prevLabel, score)          -- row per cell
MR(token, score ARRAY[])                    -- array per token
```

where `φ[y_t, y_{t-1} | x_t] = Σ_k λ_k f_k(y_t, y_{t-1}, x_t)` and the
array form indexes as `prevLabel * |Y| + label`.  The array form is the one
that performs (§6).  **The dissertation never gives the equivalent
concrete schema for FOBN factors, stripes, or mappings.**  That is a real
gap in the published record, and if you were reimplementing this you would
be designing that schema yourself.

## 3. How imputation actually works, end to end

There is no `IMPUTE` verb.  This matters, so let me be exact about what the
user writes and what comes back.

**The one documented SQL surface for "give me the posterior over the
missing values"** is the VLDB paper's §4.5 experiment:

```sql
SELECT * FROM Sensor WHERE L = 'Dark' INFER joint-distr
```

Per the paper's query semantics this "will produce a new incomplete
relation Sensor' and model M_FOBN'."  Read that twice: the result of the
query is **not a table of filled-in values**.  It is another
`<R', F'>` pair - a smaller incomplete relation plus a conditioned model.
The answer to "what is probably in this cell" is an object you then query.

**The surface that returns values** is in the dissertation's Chapters 6-7,
for the information-extraction instance.  Two families, and the user picks:

```sql
-- family 1: the maximum-likelihood ("ML") world, i.e. MAP
CREATE VIEW entityTbl1-ML AS
  SELECT *, rank() OVER (ORDER BY prob(*) DESC) r
  FROM entityTbl1 WHERE r = 1;

-- family 2: top-k over the full distribution, with an optional threshold
SELECT *, rank() OVER (ORDER BY prob(*) DESC) r
FROM SQLQuery
WHERE r <= k [ AND prob(*) > threshold ];
```

and inference appears as a named operator in the select list:

```sql
SELECT Marginal(T1.docID, T1.pos)       -- distribution of the label at each position
SELECT Top-k(T1.docID, T2.docID, exist) -- top-k joint extractions satisfying a join
FROM TokenTbl1 T1, TokenTbl2 T2
WHERE T1.label = T2.label AND T1.token = T2.token AND T1.label = 'person'
GROUP BY T1.docID, T2.docID HAVING ...
```

So the engine returns **whichever the user asked for**: a single MAP value
(`r = 1` over the ML view), a ranked top-k with probabilities, or a
marginal distribution.  The system never silently picks one, and the
dissertation's central quality claim is precisely that picking MAP
silently is the mistake everyone else was making.  Quoting the conclusion:
"querying over distributions rather than the top-1 highest-probability
results can reduce false negatives (i.e., missing results) by as much as
80%."

The pipeline for one imputation query:

1. **Rewrite the predicate against the model** (`SELECT-MODEL`).  Add
   first-order factors over `Exist^p` so the model knows the query made
   tuple existence uncertain.
2. **Rewrite the predicate against the data** (`SELECT-DATA`).  Keep every
   tuple that satisfies the predicate *or* has a missing value in the
   predicate attribute, then take the transitive closure over the tuple
   correlation graph to retain correlated evidence tuples.
3. **Prune** with the two optimisations in §4.
4. **Infer** on what survives - ground the FOBN to a flat BN and run
   variable elimination, or run first-order inference directly on the
   FOBN, or in the IE case run Viterbi/sum-product/MCMC in SQL.
5. **Return** MAP, top-k, or marginals per the query.

Step 2 is the subtle one and the paper's clearest idea.  Naive filtering is
*wrong*, for a reason worth stating in full because it is the thing a
reimplementation gets wrong first.  Their example: an exterior sensor and a
heating-duct sensor are anti-correlated.  A query for sensors detecting
heat must *retain the Cold reading from the exterior sensor*, because that
reading is the evidence that makes the duct sensor's missing value probably
Hot.  Throwing away a tuple that fails the predicate silently corrupts the
inference that depends on it.  So selection must be **sufficient** (keep
all relevant evidence) and **minimal** (drop everything else), and the
paper notes the truly minimal answer requires computing the marginal over
every `Exist^p`, which is NP-hard.  Everything in §4 is an approximation
to that.

## 4. Operators and the SQL/not-SQL split

Every relational operator is **two** algorithms, one over the model and one
over the data.

**Selection.**  Over the model: conditioning is the wrong operation.
Conditioning and marginalisation cannot express possible worlds with
*different numbers of tuples*, and selection obviously can.  Hence the
`Exist^p` attribute, and `SELECT-MODEL` adds two factors - one making
`Exist^p` depend on the predicate's probabilistic attribute over the
predicate's entity set (`CPT: Exist^p = 1 iff A^p Θ ct`), and one setting
`Exist^p = 0` over the residual entity set.  Over the data: the transitive
closure described above, plus

* **Evidence-based early stopping** - run **Bayes Ball** to find tuples
  that are conditionally independent given the evidence, and stop chasing
  there.  A vertical correlation does not imply relevance.
* **Factor-based data filtering** - use the local CPT to show
  `Pr[r | Y_ev, ρ.A^p = ct] = 0` and drop the tuple outright.  Deliberately
  *local*: they condition only on the same tuple's evidence, not the whole
  model, because full inference is the expensive thing they are trying to
  avoid.

**Projection.**  Must retain attributes not in the projection list if a
factor correlates them with attributes that are, then delete only the
factors that mention nothing surviving.  **Duplicate elimination is left
unsolved** and the reason is structural: it "is inherently not a
first-order operation," because accounting for duplicates forces the model
down to tuple level, which destroys the compression the whole design is
built on.

**Join.**  `R1 ⋈_ρ R2 = σ_ρ(R1 × R2)`, so join reduces to probabilistic
selection over the cross product.  Model side adds an existence factor
whose child stripe is over `π_{K ∪ K'}(R1 × R2)` with *two* parent stripes
(the two join attributes) and `CPT: Exist^p = 1 iff ρ`.  Stripes get
renamed first so attribute references stay unique.

**Inference.**  Two routes, named explicitly: ground the FOBN to a flat BN
and use textbook algorithms, or run first-order inference directly, adapted
from Poole (IJCAI 2003) - instead of multiplying identical CPTs once per
instance, raise the shared CPT to the power of the population size.

**What ran in SQL and what did not.**  This changed between the two
papers, and it is the most useful thing here for an implementer:

* **VLDB 2008**: PostgreSQL 8.2.4 for the relational side; inference in
  **Kevin Murphy's BNT toolbox, outside the database**.  The paper admits
  it: "Due to time limitations, rather than integrating an inference
  operator in our prototype, we used the implementation of the Variable
  Elimination algorithm from Kevin Murphy's BNT toolbox."  The 2008
  prototype supported selection and limited first-order inference and
  nothing else.
* **Dissertation, 2011**: Viterbi, sum-product, MCMC Metropolis-Hastings,
  and MCMC Gibbs **all implemented inside PostgreSQL 8.4**, in SQL.  The
  8.4 dependency is not incidental - they needed `WITH RECURSIVE` for the
  dynamic program and window functions for the per-step top-1, both of
  which landed in 8.4.

The three Viterbi implementations are instructive.  `ViterbiAllStr` does
every document at once and generates an intermediate table so large it does
not scale.  `ViterbiPerStr` does one string at a time in a recursive CTE
and is 6x slower than hand-written Java.  `ViterbiArray` keeps the factor
table as a PostgreSQL array and adds a C UDF (`top1-array`) for the
per-step argmax, and matches Java.  The conclusion they draw, stated as a
guideline, is that "tables and rows are heavy-weight representations for
cells in a factor table" and array types with a small C UDF are the right
middle ground.  Pushing predicates *into* the dynamic program is the other
half: `SELVITERBI` carries a condition `{i, len, y'}` through the V-matrix
recursion and aborts the whole string at the first "pruning position"
where no surviving partial path backtracks to the required cell.

## 5. Cost and scalability, with the real numbers

**Selection and inference (VLDB 2008 §4; hardware: 2.8 GHz CPU, 1 GB RAM,
Fedora Core 5, PostgreSQL 8.2.4).**  Synthetic hotel sensornet, 10 rooms x
8 sensors, data forward-sampled from the model.  Four algorithms:
`PlainSel` (closure, no optimisation), `FactorSel` (+ factor filtering),
`EvidenceSel` (+ Bayes Ball), `FullSel` (both).  At `size`=10,000,
`sel`=0.01, `cratio`=7/8, `mratio`=0.01:

| | PlainSel | FactorSel | EvidenceSel | FullSel |
|---|---|---|---|---|
| tuples handed to inference | 1512 | 1184 | 204.5 | 159 |
| inference, grounded BN (s) | 27.1 | 21.9 | 5.73 | 5.53 |
| inference, first-order (s) | 2.0 | 1.6 | 1.1 | 0.9 |

Selection itself runs 0 to 3.5 s across relation sizes 20k-100k.  The
headline ratio: "The increased cost for the select operation is more than
ten times less than the reduced cost for the inference operation."  That is
the economic argument for the whole paper - spend a little in the relational
engine to avoid a lot in the inference engine.

Two honest caveats they report themselves.  `FactorSel` *reduces output
size but increases execution time*.  And the early-stopping algorithms
degrade as `mratio` rises, because they run on evidence and a high missing
ratio means less evidence; at the high end `EvidenceSel` costs more than
doing nothing.  The first-order numbers are labelled **best case** - every
room shares one model, so the sharing is maximal.  "First-order inference
does not always reduce inference runtime.  In fact, there are cases where
the overhead can outweigh the improvement."

**In-database Viterbi (dissertation §5; 2.4 GHz Pentium 4, 1 GB RAM,
PostgreSQL 8.4-dev).**  200k yellow-pages address strings (6.8 tokens, 8
labels) and 18k bibliography entries (37.8 tokens, 27 labels).  Average
inference time per string:

| dataset | ViterbiAllStr | ViterbiPerStr | ViterbiArray | Java |
|---|---|---|---|---|
| address | 10.5 ms | 3.2 ms | **0.8 ms** | 0.5 ms |
| bib | 1760.1 ms | 175.1 ms | **6.2 ms** | 16.2 ms |

Averaged over the address set: 0.57 ms Java, 0.81 ms `ViterbiArray`.  On
the bibliography set **SQL beats Java**, for two stated reasons: the MR
matrix is materialised rather than recomputed, and the array join has
better memory locality when the label set is large.

**Quality, which is the real payoff.**  Probabilistic selection versus
selection over the ML world, five label sets in the Contact dataset:
false negatives for `department` fall **to 30%**, false positives stay
**≤ 0.01**, and even at a threshold of 10^-6 false positives stay below
0.05.  Cost: `prob-sel` runs **2 to 5x** `sel-ML.opt`.  For joins at k=20,
false negatives go 0.373 -> 0.251 (a third) for the optimised version while
false positives go 0.082 -> 0.110; the naive version gets false negatives
to 0.191 but blows false positives up to 0.567, about 5x.  Optimised
selection over the ML world is 2-3x the naive version for 6 conditions;
optimised join-over-ML is >5x; hybrid inference reaches 10-fold.

So the whole system's claim, stated plainly: **you can recover most of the
answers that MAP-only pipelines lose, for a 2-5x runtime cost, if the model
lives in the database.**

## 6. Limitations, and what stalled it

Stated by the authors:

* **Duplicate-free projection is unsolved** and not merely unimplemented.
  It is not a first-order operation.
* **Learning the FOBN is not implemented.**  VLDB 2008: "Developing a
  complete, efficient learning solution for the BayesStore FOBN model is in
  our immediate plans."  It never appears in the dissertation, which
  implements inference only.  So in practice the Bayes net was
  hand-specified - which is exactly the framing in the question.
* **Discrete domains only.**  Continuous and hybrid RVs are "possible to
  extend" and deferred.
* **FOBN global groundability** - the structural constraints that guarantee
  a set of overlapping stripes grounds to an acyclic BN are deferred to the
  unpublished full version.
* **First-order inference only partially implemented**; "a complete
  implementation of first-order inference inside BAYESSTORE is among our
  top priorities."
* **No cost-based optimizer.**  What shipped was a heuristic chooser for a
  single model class, and the future-work chapter asks for a real one.
* **No extensible framework.**  Each PGM type (FOBN, linear-chain CRF) got
  its own hand-written operators and inference code.  The dissertation's
  own assessment: building that per model "is very time consuming," and an
  extensible architecture is future work.
* Intractability is inherited, not solved.  Minimal selection is NP-hard;
  cyclic models are NP-hard; and the dissertation notes that an
  inverted-file text representation makes even "safe plans" produce cyclic
  models.

**Did it ship?**  No.  There is no BayesStore product and I found no public
source release.  The one artifact that was released is narrow and real: a
BSD-licensed `ViterbiCRF` package for PostgreSQL 8.4.1 (C UDF
`topk_array.c` plus `viterbi_top1.sql`, with an Enron signature sample
dataset), last edited 2010-02-01, recovered via the Wayback Machine from
`cs.berkeley.edu/~daisyw/ViterbiCRF.html`.

**Where the code went.**  That package is cited by name as reference [4] in
**Apache MADlib's** `crf.sql_in` - "D. Wang, ViterbiCRF,
http://www.cs.berkeley.edu/~daisyw/ViterbiCRF.html".  MADlib's `crf`
module (`vcrf_top1_label` as a C function, `vcrf_top1_table` in
PL/Python) is the direct descendant: in-database CRF training plus Viterbi
returning the best label sequence *and its conditional probability*.
Hellerstein, Ré, Wang et al., "The MADlib Analytics Library, or MAD Skills,
the SQL," PVLDB 5(12), 2012, doi 10.14778/2367502.2367510.

That descendant is now dead too.  **MADlib became an Apache TLP in July
2017 and retired in September 2026; it is moving to the Apache Attic**
(<https://attic.apache.org/projects/madlib.html>).  Last release 2.1.0,
8 September 2023.  Greenplum, its commercial host, went to EMC in 2010 and
Alteryx in 2022 (per Hellerstein's own page).  The source survives
read-only at gitbox.apache.org; the `crf` and `bayes` modules are still in
the tree.

Wang moved to the University of Florida and continued the line as
probabilistic *knowledge bases* rather than probabilistic databases:
"Knowledge Expansion over Probabilistic Knowledge Bases" (SIGMOD 2014),
"ScaLeKB" (VLDB Journal 2016), "ArchimedesOne" (PVLDB 2016).  Berkeley's
own thread turned into MADlib and then into the systems-for-ML work that
has nothing to do with graphical models.

So: the idea did not fail on its merits, it was overtaken.  Nothing in the
published record says anyone found FOBNs wrong.  What happened is that the
two hard parts the authors flagged - learning the structure, and an
extensible framework - were never done, and by the time anyone wanted
"fill in the missing value" again, the answer was a learned model that
needed no structure specified at all.

## 7. Related work worth knowing

**The probabilistic-database canon.**  Suciu, Olteanu, Ré, and Koch,
*Probabilistic Databases*, Synthesis Lectures on Data Management, 2011,
doi 10.1007/978-3-031-01879-4.  The theory that frames all of it: possible
worlds semantics, the extensional/intensional split, safe plans, and
Dalvi and Suciu's result that query evaluation in a tuple-independent PDB
is **#P-complete** in the size of the database modulo safe plans.  The
dissertation's §2.4.1 connects the two literatures cleanly: safe plans
correspond to tree-structured graphical models where inference is easy, and
queries needing intensional evaluation correspond to cyclic models where it
is NP-hard.  That equivalence (via Sen and Deshpande) is the single most
useful idea to carry out of this whole area.

**Contemporaries, with institutions confirmed via OpenAlex/Crossref
affiliation data:**

| system | who | where | model |
|---|---|---|---|
| MystiQ | Boulos, Dalvi, Mandhani, Mathur, Ré, Suciu | Washington | tuple-independent, extensional |
| Trio / ULDB | Benjelloun, Sarma, Halevy, Widom (2006) | Stanford | X-relations + lineage, intensional |
| Orion / U-DBMS | Cheng, Singh, Prabhakar; Singh, Mayfield et al. | Purdue | continuous intervals, intra-tuple only |
| PrDB | Sen and Deshpande (ICDE 2007) | Maryland | graphical model, one RV per tuple |
| MayBMS | Antova, Jansen, Koch, Olteanu | Cornell/Saarland | U-relations |
| MCDB | Jampani, Xu, Wu, Perez, Jermaine; Haas | Florida + IBM Almaden | Monte Carlo, VG functions |
| Lahar | Letchner, Ré, Balazinska, Philipose | Washington + Intel Seattle | Markovian streams |
| Tuffy | Niu, Ré, Doan, Shavlik (2011) | Wisconsin | Markov logic in an RDBMS |
| DeepDive | Ré et al. (2016) | Stanford | factor graphs, KBC |

PrDB is the closest contemporary and the dissertation says so, crediting it
with the same approach but arguing BayesStore delivered an integrated
system with joint optimisation where PrDB delivered individual techniques.

**The revival with learned models instead of hand-built Bayes nets.**
This is the branch that matters for your question.

* **HoloClean** - Rekatsinas, Chu, Ilyas, Ré, PVLDB 2017,
  doi 10.14778/3137628.3137631, arXiv 1702.00820.  The honest heir: it
  compiles a dirty dataset into a **probabilistic program over a factor
  graph** and infers repairs, reporting ~90% precision and >76% recall,
  2x F1 over prior art.  Same mathematics as BayesStore, with the model
  *generated* rather than specified.
* **"Can Foundation Models Wrangle Your Data?"** - Narayan, Chami, Orr,
  Ré, PVLDB 16(4), 2022, doi 10.14778/3574245.3574258, arXiv 2205.09911.
  This is the paper that makes the question concrete.  They cast
  imputation as prompting and beat the specialist systems:

  | imputation accuracy | Restaurant | Buy |
  |---|---|---|
  | HoloClean | 33.1 | 16.2 |
  | IMP (fine-tuned RoBERTa) | 77.2 | 96.5 |
  | GPT-3-175B zero-shot | 70.9 | 84.6 |
  | GPT-3-175B, k=10 | **88.4** | **98.5** |

  Note what the comparison is: 10 in-context examples beat a fine-tuned
  model and crush the factor-graph system.  And note what it is *not*: an
  accuracy number, not a calibration number.  They never claim the model's
  confidence is trustworthy.
* **Mei, Song, Fang, Yang, Fang, Long, "Capturing Semantics for Imputation
  with Pre-trained Language Models," ICDE 2021**, doi
  10.1109/icde51399.2021.00013.  The BERT-era version of the same move.
* **Table-GPT** (Li et al., SIGMOD 2024, doi 10.1145/3654979) for the
  table-tuned variant, and **LOTUS** (Patel, Jha, Pan, Gupta, Asawa,
  Guestrin, Zaharia; PVLDB 2025, doi 10.14778/3749646.3749685, Stanford and
  Berkeley) for semantic operators in a relational algebra **with accuracy
  guarantees** - which is the closest thing today to BayesStore's project
  of making a statistical model a first-class query citizen, and it is
  again partly Berkeley.

**Calibration, which is the hinge of §8.**  Guo, Pleiss, Sun, Weinberger,
"On Calibration of Modern Neural Networks," ICML 2017, arXiv 1706.04599.
Modern networks are poorly calibrated; **temperature scaling, a single
parameter**, fixes most of it.  Also Desai and Durrett, "Calibration of
Pre-trained Transformers," EMNLP 2020, doi 10.18653/v1/2020.emnlp-main.21,
and Kadavath et al., "Language Models (Mostly) Know What They Know,"
arXiv 2207.05221.

## 8. The analogous feature in pg_infer, and whether the analogy holds

### What genuinely aligns

Four things, and they are not trivial.

**The model is a first-class database object stored beside the relations.**
BayesStore's `DB^p = <R, F>`, with F deliberately *not* attached to tuples
and deliberately exposed so it "can be queried and manipulated by
users/applications," is structurally what a pg_infer vindex is:
`infer_create_model('qwen05b', '/data/qwen-0.5b.vindex')` registers a
model by name, it is read-only at query time, and `describe`, `walk`,
`infer_show_features` query it directly.  This is the deepest alignment and
it is real.  pg_infer has already made the architectural choice that was
the 2008 paper's main argument.

**Model-side and data-side operators are distinct.**  BayesStore's
`σ(M_FOBN)` versus `σ(R)` split, and the pushing of predicates into
inference (`SELVITERBI`), has an existing counterpart: pg_infer's index
access method and the `<~>` operator already push a model query into a
PostgreSQL plan, and `am_cost.rs` already means the planner costs it.  The
machinery for "evaluate part of this inside the engine" exists.

**Query-driven inference beats batch inference.**  The dissertation's
framing of the problem is that the loose-coupling practice exports
everything, runs the model over all of it, and imports the top-1 back,
losing both time and information.  pg_infer's `describe_many` and
`similar_to_many` batching into one round trip is the same instinct, and
the one-round-trip framing in `fn_similar.rs` is exactly the economic
argument BayesStore made.

**Returning a confidence alongside the value.**  BayesStore returns
`prob(*)`; `describe()` returns a `confidence` column.  Same column in the
same place in the result.

### Where it breaks

**1. `confidence` is not a probability, and the code already knows it.**
`fn_describe.rs:156` maps `e.gate_score` straight into the column named
`confidence`.  A gate score is `gemv(gate_matrix_row, mean_pooled_embedding)`
(`crates/infer-vindex/src/index/gate.rs`) - an unnormalised dot product.
It does not sum to one over anything, there is no reference class, and the
scale is model-dependent: the project's own BitNet qualification records
`Zone 332.80` as a top edge while the README's Gemma-class examples sit at
20-45.  `similar_to_impl` is worse for this purpose - it returns
`max_shared_score` when gate features overlap and falls back to
`cosine * 10.0` when they do not, so the same output column carries two
incommensurable quantities depending on which branch fired.  Imputation
needs `P(value | evidence)`.  A gate score cannot be coerced into one by
rescaling, because there is nothing it is a monotone transform *of*.

**2. There is no conditional independence structure, and both of
BayesStore's optimisations are consequences of having one.**  This is the
part people skip.  A Bayes net's value is not that it gives you numbers; it
is that the factorisation tells you *which variables screen off which*.
That single fact is what makes `EvidenceSel` possible (Bayes Ball proves a
correlated tuple irrelevant given the evidence, so stop chasing it) and
what makes `FactorSel` possible (a local CPT proves a value impossible, so
drop the tuple).  Those two optimisations are the difference between 27.1 s
and 5.53 s in their own table.  pg_infer has no graph over columns at all.
Gate co-activation is a similarity relation: symmetric, unnormalised, with
no d-separation semantics.  So there is no analogue of either optimisation,
and this is not an implementation gap to be filled later - it is a
different model class.

**3. No possible-worlds semantics, so no closure, so no composition.**
BayesStore's operators are closed: `σ`, `π`, `⋈` each take `<R, F>` to a
valid `<R', F'>`, which is what lets you write a join over two imputed
relations and still get a meaningful probability out.  That closure is the
reason the paper bothers with `Exist^p` and existence factors at all.  If
pg_infer imputes two columns of the same row, there is no joint
distribution over the two answers, no correlation between them, and no way
to answer "what is the probability both are right."  You would have two
scores and no composition rule.

**4. No domain, and the relation labels are a hardcoded heuristic.**  A
BayesStore probabilistic attribute has `dom(A_i)` and its CPT is a
distribution *over that domain*.  `describe()` returns whatever tokens the
activated gate features happen to be labelled with; there is no closed
candidate set and no guarantee the answer is even a legal value for the
column.  Worse for the analogy, the `relation` column is not learned at
all: `relation_classify.rs` decides it from three layer bands, target
capitalisation, two magic thresholds (20.0 and 30.0), and a hardcoded
twenty-word English geography list.  The project's own
`bitnet-qualification.md` documents it failing off-distribution -
`describe('France')` returning `Zone`, `demographics`, `Barb`, which the
doc correctly calls "structurally correct output ... but weak semantics."
A CPT is a measured object; this is a rule of thumb tuned to one model
family.

**5. The model was never fitted to this database.**  A BayesStore CPT comes
from this table (or from an expert who knows this domain).  A transformer's
weights come from a pretraining corpus and have never seen your schema.
For "fill in the city given the street and the postcode," a Bayes net
fitted to the table is right by construction on the rows it was fitted to,
where the transformer is reciting world knowledge.  This cuts both ways -
the FM paper's central finding is that pretrained models impute values
*absent from the training set*, which the CPT cannot do at all - but it
means the number you report is not a statement about your data.  That is a
semantic difference, not a quality difference, and it is the one most
likely to mislead a user who sees a column called `confidence`.

**6. pg_infer's distinctive mechanism is the wrong mechanism for this.**
The project's stated selling point is answering questions about a model
"without running inference," by reading weights.  But imputation *is* an
inference problem: you have evidence and you want a posterior.  The one
function in pg_infer that produces a genuine normalised distribution is
`infer()`, which runs a real forward pass and a real softmax
(`crates/infer-inference/src/forward/predict.rs::logits_to_predictions`:
logits, optional softcap, temperature, max-subtracted exp, divide by the
sum).  Those `probability` values are the only normalised probabilities in
the extension.  So the sound version of this feature routes through
`infer()` and not through `describe()` - which means the analogy points
*away* from gate vectors, the thing that makes pg_infer pg_infer, and
toward ordinary LLM imputation that any extension with an HTTP client
could do.

### Is the analogy sound?

**At the level it is usually meant - "use the transformer's learned
associations the way BayesStore used a Bayes net" - it is superficial.**
The two systems agree on the architecture (model as a first-class,
separately-stored, queryable database object, with inference driven by the
query) and disagree on the only thing imputation needs: BayesStore has a
normalised joint distribution with declared conditional independence, over
a declared domain, fitted to the data in question.  pg_infer's gate scores
have none of those four properties.  Objections 1 through 4 are not a
backlog, they are the definition of the difference.

**At one narrower level it is sound, and worth taking.**  The three things
to borrow are architectural, not probabilistic: (a) the `<R, F>`
separation, which pg_infer already has; (b) the discipline of making the
user *choose* MAP versus distribution rather than silently returning
top-1 - BayesStore's `Top-k` versus `Marginal`, and their measurement that
silent MAP loses up to 80% of true answers, is the single most transferable
result in this report; (c) query-driven rather than batch inference.

### The minimum honest version, if you want to build it

Four steps, in order, and the third is the one that does the work.

1. **Candidate generation from gate scores.**  This is what `describe()`
   and `similar_to()` are legitimately good for - proposing which values
   are plausible at all.  Report the score as a score.  Never call it a
   probability.
2. **A declared domain.**  The function signature needs the candidate set:
   either an explicit list, or `SELECT DISTINCT col` over the non-null
   rows, or an enum/FK.  Without a domain there is nothing to normalise
   over, and §8.4 stays broken no matter what else you do.
3. **Normalise with a real forward pass.**  Build a prompt from the row's
   non-null columns, run `infer()`, and **renormalise the softmax over
   only the token sequences spelling the declared candidates**.  That step
   is what converts an unnormalised score into `P(value | row, domain)`.
   It is the step that makes the BayesStore analogy true rather than
   rhetorical, and skipping it means the feature is a ranking with a
   misleading column name.
4. **Then calibrate.**  Fit one temperature parameter per (table, column)
   against a held-out sample of rows where the value is known
   (Guo et al. 2017 - one parameter, and it is most of the fix).  Store it
   with the model registration.  Without this the probability is normalised
   but not calibrated, and "calibrated" is what the request asked for.

API shape that keeps BayesStore's distinction honest:

```sql
-- the distribution (BayesStore's Marginal)
SELECT * FROM impute('cities', 'country', row_to_json(c))  -- value, probability
  FROM cities c WHERE c.country IS NULL;

-- the MAP value and nothing else (BayesStore's ML view)
SELECT impute_map('cities', 'country', row_to_json(c)) FROM ...;
```

Be explicit in the docs about the three things you cannot offer, because
each is a question a user will ask: no joint over several imputed columns,
no pruning by conditional independence, no closure under joins.  And the
honest framing of what you *do* offer is "a calibrated conditional over a
declared domain, from a model that has never seen your data" - which is a
genuinely useful thing, and a different thing from what BayesStore did.

## 9. What I could not verify

* **The "full version" of the VLDB 2008 paper.**  Referenced four times,
  not publicly available that I can find.  The FOBN global acyclicity
  constraints and the join special case exist only there.
* **The concrete relational schema for FOBN factors, stripes, and
  mappings.**  Given for the CRF instance (the MR matrix), never for FOBN
  in either source.
* **Whether the `INFER joint-distr` syntax was ever more than the one
  example.**  The dissertation uses a different surface
  (`Top-k(...)`/`Marginal(...)` in the select list) and never mentions
  `INFER`.  I suspect the SQL extension was sketched and dropped, but the
  record does not say so.
* **Any public BayesStore source release.**  None found.  Only the
  `ViterbiCRF` package, recovered from the Wayback Machine.
* **MauveDB's full text.**  Every PDF mirror I tried was dead; I worked
  from the SIGMOD 2006 abstract and metadata.  Enough to confirm the
  institutions and that its models are regression and interpolation views
  over Apache Derby, not Bayes nets - so, ruled out - but I have not read
  the paper.
* **BBQ's official VLDB PDF.**  vldb.org's 2004 proceedings links are
  dead; I read the authors' copy at `cs.umd.edu/~amol/papers/vldb04.pdf`,
  which has the matching title, author list, and affiliations.
* **MADlib's exact retirement date.**  The Attic page says "retired in
  September 2026 and the move to the Attic is In Progress" without a day.
