# ProtEng-TermBench

A benchmark for AI agents doing protein engineering and early drug discovery
in a terminal. Six tasks follow one programme from a predicted structure to a
characterised inhibitor. Each task is generated from a hidden seed, solved by
an expert reference and graded by code that checks the answer against planted
truth and checks that the method was sound.

* [Validation dashboard](dashboard.md): reference and wrong-method pass rates, from the recorded campaigns
* [Design](DESIGN.md): the six tasks, the traps and the three grading layers
* [Requirements](REQUIREMENTS.md)
* [Data](DATA.md): sources and licences
* [Validation report](VALIDATION_REPORT.md): thresholds, flaws found and fixed

## Try it

```bash
git clone https://github.com/mbote-droid/proteng-termbench && cd proteng-termbench
uv sync --frozen --group agent
uv run proteng grade t05-qsar-scaffold-hop \
    --submission tasks/t05-qsar-scaffold-hop/solution/qsar_model.py \
    --seeds tasks/t05-qsar-scaffold-hop/tests/dev_seeds.txt \
    --runner local --trusted
```

Or open it in a [Codespace](https://codespaces.new/mbote-droid/proteng-termbench)
and run the same command there.

Source and issues: [github.com/mbote-droid/proteng-termbench](https://github.com/mbote-droid/proteng-termbench).
