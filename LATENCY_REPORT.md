# Synthetic live latency sample

The subsequent SQL correctness investigation and matched before/after review
results are documented in [SQL_REVIEW_INVESTIGATION.md](SQL_REVIEW_INVESTIGATION.md).
The measurements below are the original baseline and have not been rewritten.

Twenty live requests, five per category, used a temporary SQLite fixture with 36 monthly periods and two regions. The fixture was deleted after the run. No production database was queried. Times are seconds relative to each request's start and are rounded to four decimals. Stage duration sums are not end-to-end latency because calls can overlap. A measured gap includes graph work and other code outside the instrumented spans; it is not an attributed bottleneck.

The bounded sample made 107 model calls, with 143,743 reported input tokens and 15,289 output tokens. Estimated API cost was $0.027538 for these 20 requests. An initial sandboxed connection attempt failed before this sample and is excluded. Outcomes were 5 `passed`, 13 `data_only`, and 2 `blocked`. The fixture, network conditions, and small sample limit how far these timings can be generalized.

| Category | Requests | Median wall time | Slowest wall time | Outcomes | Median measured gap |
|---|---:|---:|---:|---|---:|
| lookup | 5 | 14.24 | 15.29 | blocked 2, data_only 2, passed 1 | 0.01 |
| aggregation | 5 | 10.76 | 15.64 | data_only 1, passed 4 | 0.01 |
| forecast_single | 5 | 22.08 | 27.73 | data_only 5 | 0.05 |
| forecast_grouped | 5 | 21.69 | 25.53 | data_only 5 | 0.01 |

Stage time below is the mean total duration *per request* in each category, including retries. It is not additive across stages. A zero means that stage did not run.

| Category | sql | sql_critic | database | forecast_fit | forecast_backtest | summary | answer_critic | chart |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| lookup | 2.47 | 9.85 | 0.00 | 0.00 | 0.00 | 1.06 | 0.24 | 0.97 |
| aggregation | 1.38 | 1.90 | 0.00 | 0.00 | 0.00 | 5.46 | 2.41 | 3.99 |
| forecast_single | 0.00 | 0.00 | 0.00 | 1.17 | 3.25 | 16.43 | 0.76 | 0.00 |
| forecast_grouped | 2.71 | 17.56 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

## Findings and limits

- Lookup latency was dominated by repeated SQL critic calls. All five lookup requests needed two SQL model attempts; several reviews made a second internal verdict-repair call. Only one lookup passed. This sample cannot isolate a normal successful lookup path well.
- Aggregations had four passed answers. Summary generation was the largest sequential measured stage on average (5.46s per request), followed by answer review (2.41s). The chart averaged 3.99s but started alongside the summary and was largely hidden by summary and review work. Its duration cannot be added to those stages to estimate wall time.
- Single-series forecast requests used deterministic history SQL and reached fitting and backtesting. Fitting averaged 1.17s and backtesting 3.25s per request, while three summary calls per request averaged 16.43s in total. All five ended `data_only`; three did not reach an LLM answer-critic call, so these are measurements of a failed-review path rather than successful forecast answers. Two requests re-executed SQL and refit after an answer-review visit.
- Grouped forecasts were dominated by SQL critic calls (17.56s per request on average). Every request exhausted SQL review and returned `data_only`; **none reached forecast fitting or backtesting**. This sample therefore cannot establish grouped forecasting or grouped backtest latency. The table's zeroes mean the stages did not run, not that they are free.
- SQLite execution was under 0.01s per call on this tiny fixture. That does not predict execution time on production-scale data or a Windows bind mount. The small measured gaps reflect this sample only; they do not prove that all application overhead is negligible in other environments.
- No latency optimization or retry-budget change was made. The review outcomes suggest a separate correctness investigation before optimizing the failed paths. Studying recurring question patterns would require a separate, explicitly approved question sample; request_metrics contains no question text.


## Request timelines

Each line gives a stage's start–end interval. `chart*` ran in the background worker; intervals that intersect another stage show actual overlap. Repeated stage names show additional attempts or graph revisits. SQL model attempts, summary visits, database executions, and forecast fits are listed to expose retries without guessing their cause from telemetry alone.

### lookup 1

Wall 14.24s; data_only; 6 model calls; 5228 input / 467 output tokens; estimated API cost $0.001064. Measured gap 0.02s; chart/summary overlap 0.00s.

Attempts/visits: SQL model 2, SQL critic 4, database 1, forecast fit 0, backtest 0, summary 0, answer critic 0.

Retry evidence: SQL generation 1; summary regeneration 0; database re-execution 0; forecast self-retry 0. Multiple SQL critic calls for one SQL attempt can reflect verdict repair. The timeline shows order; telemetry does not carry a definitive retry-cause code.

```text
  0.0066–  1.9088  sql 
  1.9101–  5.1611  sql_critic 
  5.1617–  7.9912  sql_critic 
  7.9927–  9.2490  sql 
  9.2502– 11.4873  sql_critic 
 11.4875– 14.2291  sql_critic 
 14.2345– 14.2364  database 
```

### lookup 2

Wall 15.29s; passed; 7 model calls; 7203 input / 895 output tokens; estimated API cost $0.001617. Measured gap 0.02s; chart/summary overlap 4.86s.

Attempts/visits: SQL model 2, SQL critic 2, database 1, forecast fit 0, backtest 0, summary 1, answer critic 1.

Retry evidence: SQL generation 1; summary regeneration 0; database re-execution 0; forecast self-retry 0. Multiple SQL critic calls for one SQL attempt can reflect verdict repair. The timeline shows order; telemetry does not carry a definitive retry-cause code.

```text
  0.0019–  1.0792  sql 
  1.0820–  4.3671  sql_critic 
  4.3689–  5.7142  sql 
  5.7155–  8.8049  sql_critic 
  8.8060–  8.8075  database 
  8.8158– 13.6821  chart*
  8.8255– 14.1022  summary 
 14.1045– 15.2826  answer_critic 
```

### lookup 3

Wall 14.95s; data_only; 6 model calls; 5262 input / 529 output tokens; estimated API cost $0.001107. Measured gap 0.01s; chart/summary overlap 0.00s.

Attempts/visits: SQL model 2, SQL critic 4, database 1, forecast fit 0, backtest 0, summary 0, answer critic 0.

Retry evidence: SQL generation 1; summary regeneration 0; database re-execution 0; forecast self-retry 0. Multiple SQL critic calls for one SQL attempt can reflect verdict repair. The timeline shows order; telemetry does not carry a definitive retry-cause code.

```text
  0.0018–  1.2278  sql 
  1.2290–  5.4304  sql_critic 
  5.4306–  9.3094  sql_critic 
  9.3121– 10.5942  sql 
 10.5954– 12.2315  sql_critic 
 12.2316– 14.9435  sql_critic 
 14.9476– 14.9495  database 
```

### lookup 4

Wall 12.32s; blocked; 6 model calls; 5337 input / 490 output tokens; estimated API cost $0.001095. Measured gap 0.01s; chart/summary overlap 0.00s.

Attempts/visits: SQL model 2, SQL critic 4, database 0, forecast fit 0, backtest 0, summary 0, answer critic 0.

Retry evidence: SQL generation 1; summary regeneration 0; database re-execution 0; forecast self-retry 0. Multiple SQL critic calls for one SQL attempt can reflect verdict repair. The timeline shows order; telemetry does not carry a definitive retry-cause code.

```text
  0.0030–  1.3553  sql 
  1.3562–  5.6251  sql_critic 
  5.6253–  7.6203  sql_critic 
  7.6217–  8.6774  sql 
  8.6786– 10.4094  sql_critic 
 10.4096– 12.3166  sql_critic 
```

### lookup 5

Wall 11.37s; blocked; 6 model calls; 5148 input / 479 output tokens; estimated API cost $0.001060. Measured gap 0.01s; chart/summary overlap 0.00s.

Attempts/visits: SQL model 2, SQL critic 4, database 0, forecast fit 0, backtest 0, summary 0, answer critic 0.

Retry evidence: SQL generation 1; summary regeneration 0; database re-execution 0; forecast self-retry 0. Multiple SQL critic calls for one SQL attempt can reflect verdict repair. The timeline shows order; telemetry does not carry a definitive retry-cause code.

```text
  0.0012–  0.9736  sql 
  0.9742–  2.7879  sql_critic 
  2.7881–  5.2422  sql_critic 
  5.2463–  6.1342  sql 
  6.1352–  8.2380  sql_critic 
  8.2383– 11.3664  sql_critic 
```

### aggregation 1

Wall 7.61s; passed; 5 model calls; 3742 input / 477 output tokens; estimated API cost $0.000847. Measured gap 0.01s; chart/summary overlap 1.91s.

Attempts/visits: SQL model 1, SQL critic 1, database 1, forecast fit 0, backtest 0, summary 1, answer critic 1.

Retry evidence: SQL generation 0; summary regeneration 0; database re-execution 0; forecast self-retry 0. Multiple SQL critic calls for one SQL attempt can reflect verdict repair. The timeline shows order; telemetry does not carry a definitive retry-cause code.

```text
  0.0022–  1.6323  sql 
  1.6333–  3.6228  sql_critic 
  3.6237–  3.6247  database 
  3.6253–  6.9935  chart*
  3.6271–  5.5332  summary 
  5.5346–  7.6022  answer_critic 
```

### aggregation 2

Wall 15.64s; data_only; 9 model calls; 7392 input / 892 output tokens; estimated API cost $0.001644. Measured gap 0.01s; chart/summary overlap 2.37s.

Attempts/visits: SQL model 1, SQL critic 1, database 1, forecast fit 0, backtest 0, summary 3, answer critic 3.

Retry evidence: SQL generation 0; summary regeneration 2; database re-execution 0; forecast self-retry 0. Multiple SQL critic calls for one SQL attempt can reflect verdict repair. The timeline shows order; telemetry does not carry a definitive retry-cause code.

```text
  0.0021–  1.3340  sql 
  1.3352–  3.0760  sql_critic 
  3.0775–  3.0792  database 
  3.0800–  6.1931  chart*
  3.0831–  5.4534  summary 
  5.4546–  7.4159  answer_critic 
  7.4177–  9.3068  summary 
  9.3084– 11.4738  answer_critic 
 11.4759– 13.5605  summary 
 13.5618– 15.6349  answer_critic 
```

### aggregation 3

Wall 7.40s; passed; 5 model calls; 3742 input / 416 output tokens; estimated API cost $0.000811. Measured gap 0.01s; chart/summary overlap 2.77s.

Attempts/visits: SQL model 1, SQL critic 1, database 1, forecast fit 0, backtest 0, summary 1, answer critic 1.

Retry evidence: SQL generation 0; summary regeneration 0; database re-execution 0; forecast self-retry 0. Multiple SQL critic calls for one SQL attempt can reflect verdict repair. The timeline shows order; telemetry does not carry a definitive retry-cause code.

```text
  0.0018–  1.3591  sql 
  1.3601–  3.5223  sql_critic 
  3.5241–  3.5262  database 
  3.5271–  6.8591  chart*
  3.5304–  6.2996  summary 
  6.3012–  7.3960  answer_critic 
```

### aggregation 4

Wall 14.41s; passed; 5 model calls; 5098 input / 930 output tokens; estimated API cost $0.001323. Measured gap 0.01s; chart/summary overlap 4.98s.

Attempts/visits: SQL model 1, SQL critic 1, database 1, forecast fit 0, backtest 0, summary 1, answer critic 1.

Retry evidence: SQL generation 0; summary regeneration 0; database re-execution 0; forecast self-retry 0. Multiple SQL critic calls for one SQL attempt can reflect verdict repair. The timeline shows order; telemetry does not carry a definitive retry-cause code.

```text
  0.0020–  1.2113  sql 
  1.2129–  3.2067  sql_critic 
  3.2075–  3.2085  database 
  3.2090–  8.1948  chart*
  3.2109– 12.9835  summary 
 12.9845– 14.4078  answer_critic 
```

### aggregation 5

Wall 10.76s; passed; 5 model calls; 5098 input / 930 output tokens; estimated API cost $0.001323. Measured gap 0.01s; chart/summary overlap 5.13s.

Attempts/visits: SQL model 1, SQL critic 1, database 1, forecast fit 0, backtest 0, summary 1, answer critic 1.

Retry evidence: SQL generation 0; summary regeneration 0; database re-execution 0; forecast self-retry 0. Multiple SQL critic calls for one SQL attempt can reflect verdict repair. The timeline shows order; telemetry does not carry a definitive retry-cause code.

```text
  0.0019–  1.3797  sql 
  1.3809–  2.9821  sql_critic 
  2.9830–  2.9846  database 
  2.9854–  8.1220  chart*
  2.9889–  9.4964  summary 
  9.4977– 10.7540  answer_critic 
```

### forecast_single 1

Wall 16.79s; data_only; 3 model calls; 9258 input / 755 output tokens; estimated API cost $0.001400. Measured gap 0.07s; chart/summary overlap 0.00s.

Attempts/visits: SQL model 0, SQL critic 0, database 1, forecast fit 1, backtest 1, summary 3, answer critic 0.

Retry evidence: SQL generation 0; summary regeneration 2; database re-execution 0; forecast self-retry 0. Multiple SQL critic calls for one SQL attempt can reflect verdict repair. The timeline shows order; telemetry does not carry a definitive retry-cause code.

```text
  0.0182–  0.0249  database 
  0.0328–  2.6252  forecast_fit 
  2.6297–  4.8764  forecast_backtest 
  4.8896–  8.4377  summary 
  8.4481– 12.7831  summary 
 12.7950– 16.7845  summary 
```

### forecast_single 2

Wall 22.61s; data_only; 4 model calls; 13104 input / 802 output tokens; estimated API cost $0.001986. Measured gap 0.05s; chart/summary overlap 0.00s.

Attempts/visits: SQL model 0, SQL critic 0, database 2, forecast fit 2, backtest 2, summary 3, answer critic 1.

Retry evidence: SQL generation 0; summary regeneration 2; database re-execution 1; forecast self-retry 0. Multiple SQL critic calls for one SQL attempt can reflect verdict repair. The timeline shows order; telemetry does not carry a definitive retry-cause code.

```text
  0.0051–  0.0067  database 
  0.0088–  0.5398  forecast_fit 
  0.5440–  3.4297  forecast_backtest 
  3.4361–  8.3541  summary 
  8.3615– 12.2035  summary 
 12.2051– 14.1605  answer_critic 
 14.1655– 14.1677  database 
 14.1693– 14.7042  forecast_fit 
 14.7079– 16.8799  forecast_backtest 
 16.8860– 22.6067  summary 
```

### forecast_single 3

Wall 22.08s; data_only; 4 model calls; 13684 input / 816 output tokens; estimated API cost $0.002062. Measured gap 0.05s; chart/summary overlap 0.00s.

Attempts/visits: SQL model 0, SQL critic 0, database 2, forecast fit 2, backtest 2, summary 3, answer critic 1.

Retry evidence: SQL generation 0; summary regeneration 2; database re-execution 1; forecast self-retry 0. Multiple SQL critic calls for one SQL attempt can reflect verdict repair. The timeline shows order; telemetry does not carry a definitive retry-cause code.

```text
  0.0036–  0.0055  database 
  0.0071–  0.5457  forecast_fit 
  0.5494–  2.9256  forecast_backtest 
  2.9314–  9.4248  summary 
  9.4389– 13.8954  summary 
 13.8969– 15.7474  answer_critic 
 15.7492– 15.7504  database 
 15.7513– 16.2799  forecast_fit 
 16.2839– 18.2858  forecast_backtest 
 18.2924– 22.0799  summary 
```

### forecast_single 4

Wall 19.09s; data_only; 3 model calls; 10451 input / 977 output tokens; estimated API cost $0.001645. Measured gap 0.04s; chart/summary overlap 0.00s.

Attempts/visits: SQL model 0, SQL critic 0, database 1, forecast fit 1, backtest 1, summary 3, answer critic 0.

Retry evidence: SQL generation 0; summary regeneration 2; database re-execution 0; forecast self-retry 0. Multiple SQL critic calls for one SQL attempt can reflect verdict repair. The timeline shows order; telemetry does not carry a definitive retry-cause code.

```text
  0.0037–  0.0056  database 
  0.0072–  0.5054  forecast_fit 
  0.5088–  2.7295  forecast_backtest 
  2.7364–  7.1795  summary 
  7.1907– 12.1456  summary 
 12.1604– 19.0877  summary 
```

### forecast_single 5

Wall 27.73s; data_only; 3 model calls; 10858 input / 1440 output tokens; estimated API cost $0.001974. Measured gap 0.05s; chart/summary overlap 0.00s.

Attempts/visits: SQL model 0, SQL critic 0, database 1, forecast fit 1, backtest 1, summary 3, answer critic 0.

Retry evidence: SQL generation 0; summary regeneration 2; database re-execution 0; forecast self-retry 0. Multiple SQL critic calls for one SQL attempt can reflect verdict repair. The timeline shows order; telemetry does not carry a definitive retry-cause code.

```text
  0.0034–  0.0054  database 
  0.0069–  0.6297  forecast_fit 
  0.6335–  2.9704  forecast_backtest 
  2.9795–  8.6717  summary 
  8.6966– 18.0919  summary 
 18.0994– 27.7311  summary 
```

### forecast_grouped 1

Wall 12.30s; data_only; 6 model calls; 6631 input / 781 output tokens; estimated API cost $0.001367. Measured gap 0.02s; chart/summary overlap 0.00s.

Attempts/visits: SQL model 2, SQL critic 4, database 1, forecast fit 0, backtest 0, summary 0, answer critic 0.

Retry evidence: SQL generation 1; summary regeneration 0; database re-execution 0; forecast self-retry 0. Multiple SQL critic calls for one SQL attempt can reflect verdict repair. The timeline shows order; telemetry does not carry a definitive retry-cause code.

```text
  0.0020–  1.1757  sql 
  1.1776–  3.8669  sql_critic 
  3.8671–  6.2795  sql_critic 
  6.2817–  7.5243  sql 
  7.5297–  9.8209  sql_critic 
  9.8211– 12.2878  sql_critic 
 12.2900– 12.2930  database 
```

### forecast_grouped 2

Wall 25.53s; data_only; 6 model calls; 6627 input / 791 output tokens; estimated API cost $0.001296. Measured gap 0.01s; chart/summary overlap 0.00s.

Attempts/visits: SQL model 2, SQL critic 4, database 1, forecast fit 0, backtest 0, summary 0, answer critic 0.

Retry evidence: SQL generation 1; summary regeneration 0; database re-execution 0; forecast self-retry 0. Multiple SQL critic calls for one SQL attempt can reflect verdict repair. The timeline shows order; telemetry does not carry a definitive retry-cause code.

```text
  0.0018–  1.1180  sql 
  1.1188–  6.1353  sql_critic 
  6.1356– 12.4642  sql_critic 
 12.4655– 13.5592  sql 
 13.5607– 20.3700  sql_critic 
 20.3702– 25.5271  sql_critic 
 25.5293– 25.5317  database 
```

### forecast_grouped 3

Wall 24.05s; data_only; 6 model calls; 6632 input / 813 output tokens; estimated API cost $0.001310. Measured gap 0.01s; chart/summary overlap 0.00s.

Attempts/visits: SQL model 2, SQL critic 4, database 1, forecast fit 0, backtest 0, summary 0, answer critic 0.

Retry evidence: SQL generation 1; summary regeneration 0; database re-execution 0; forecast self-retry 0. Multiple SQL critic calls for one SQL attempt can reflect verdict repair. The timeline shows order; telemetry does not carry a definitive retry-cause code.

```text
  0.0019–  1.8034  sql 
  1.8043–  6.8284  sql_critic 
  6.8286– 11.9804  sql_critic 
 11.9823– 13.0995  sql 
 13.1006– 19.4606  sql_critic 
 19.4608– 24.0469  sql_critic 
 24.0491– 24.0510  database 
```

### forecast_grouped 4

Wall 21.69s; data_only; 6 model calls; 6630 input / 806 output tokens; estimated API cost $0.001305. Measured gap 0.01s; chart/summary overlap 0.00s.

Attempts/visits: SQL model 2, SQL critic 4, database 1, forecast fit 0, backtest 0, summary 0, answer critic 0.

Retry evidence: SQL generation 1; summary regeneration 0; database re-execution 0; forecast self-retry 0. Multiple SQL critic calls for one SQL attempt can reflect verdict repair. The timeline shows order; telemetry does not carry a definitive retry-cause code.

```text
  0.0012–  0.9668  sql 
  0.9678–  5.3481  sql_critic 
  5.3484–  9.2119  sql_critic 
  9.2137– 10.2392  sql 
 10.2403– 15.3975  sql_critic 
 15.3977– 21.6779  sql_critic 
 21.6807– 21.6834  database 
```

### forecast_grouped 5

Wall 17.85s; data_only; 6 model calls; 6618 input / 803 output tokens; estimated API cost $0.001302. Measured gap 0.01s; chart/summary overlap 0.00s.

Attempts/visits: SQL model 2, SQL critic 4, database 1, forecast fit 0, backtest 0, summary 0, answer critic 0.

Retry evidence: SQL generation 1; summary regeneration 0; database re-execution 0; forecast self-retry 0. Multiple SQL critic calls for one SQL attempt can reflect verdict repair. The timeline shows order; telemetry does not carry a definitive retry-cause code.

```text
  0.0020–  1.8036  sql 
  1.8042–  5.4498  sql_critic 
  5.4500–  8.5527  sql_critic 
  8.5549– 10.7540  sql 
 10.7550– 14.3247  sql_critic 
 14.3250– 17.8425  sql_critic 
 17.8450– 17.8489  database 
```
