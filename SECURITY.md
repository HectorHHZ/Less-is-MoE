# Security policy

## Reporting a vulnerability

Please open a private GitHub security advisory for vulnerabilities in the code.
Do not include credentials, private model URLs, or private datasets in a public
issue.

## Credentials and experiment artifacts

This repository does not require credentials to be stored in source files.
Authenticate with the official CLI for the service you use, or provide tokens
through process environment variables. `.env` files, private keys, checkpoints,
logs, and generated outputs are ignored by Git.

Before publishing a branch, scan both the working tree and commit history for
Hugging Face, Weights & Biases, GitHub, OpenAI, and cloud credentials. If a
credential ever appeared in a commit or shared log, remove it from history and
rotate it at the provider; deleting the visible line alone is not sufficient.

The evaluation code accepts an OpenAI API key only for optional judge-based
tasks. Prefer `OPENAI_API_KEY` in the process environment. Do not pass a secret
on a shared command line or commit it in a recipe.

MBPP and HumanEval scoring execute model-generated Python in a subprocess.
The timeout prevents hangs but is not a security sandbox. Run code-generation
benchmarks only inside a disposable container or similarly isolated worker;
never on a host that contains credentials or irreplaceable data.
