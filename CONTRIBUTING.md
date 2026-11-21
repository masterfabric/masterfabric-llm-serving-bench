# Contributing

This is a lab repository. Changes are welcome when they keep it runnable on a laptop and keep every
number in the README reproducible.

## Ground rules

- **Measured numbers only.** If a change affects results, re-run the experiment
  (`make <experiment>`), regenerate the images (`make charts`), and commit the new `results/*.json`
  with them. Never edit a results file by hand.
- **Say what was computed.** A value that is derived rather than measured (the KV buffer size, for
  example) must say so in the results file and in the README.
- **New hardware is a new results file**, not a replacement. Add the machine, server build and flags
  to the commit body.
- One logical change per commit, Conventional Commit subject, and a body that explains why.
- No secrets and no model files in git. Models live in `~/.cache/mf-models`.

## Development loop

```bash
make venv models
make prefill && make charts
make clean-servers
```

## License of contributions

By contributing you agree that your contribution is licensed under the AGPL-3.0, as described in
[LICENSE](LICENSE) and [NOTICE](NOTICE).
