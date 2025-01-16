# WebGWAS

WebGWAS is a free, hosted web app for conducting approximate [genome-wide association studies (GWAS)](https://en.wikipedia.org/wiki/Genome-wide_association_study).

[Try it out!](https://webgwas.org)

This project was developed by Michael Zietz ([@zietzm](https://github.com/zietzm)) in the [Tatonetti Lab](https://tatonettilab.org/) at [Columbia University](https://www.columbia.edu/) and [Cedars-Sinai Medical Center](https://www.cedars-sinai.edu/research-education/research/departments-institutes/computational-biomedicine.html).

## How it works

WebGWAS provides an interface that lets users define phenotypes in terms of features available in the [UK Biobank](https://www.ukbiobank.ac.uk/), such as ICD-10 codes and measurement values.
For example, a phenotype could be "hypertension" defined as "diagnosis of I10" or "systolic blood pressure above 140 mmHg".
A submitted phenotype is then approximated using linear regression against all available features, and the resulting coefficients are used to appropriately weight pre-computed feature GWAS summary statistics such that the overall results approximate GWAS summary statistics for the original, user-defined phenotype.
Because the inputs are mostly pre-computed, this computation is very fast (~10 seconds).
To protect participant privacy, we store no individual-level private health information on the server.
Instead, we store only aggregated, anonymized phenotype values.

For more information, please see our [preprint](https://doi.org/10.1101/2023.11.20.567948).

## Technical details

WebGWAS consists of three main parts, represented by the top-level directories here:

- `backend`: The backend server that performs the approximate GWAS
- `frontend`: The frontend web app that allows users to define phenotypes and run GWAS
- `deploy`: The deployment scripts for the backend and frontend

The backend is written in [Rust](https://www.rust-lang.org/).
The frontend is written in [React](https://reactjs.org/) using [Next.js](https://nextjs.org/).
Deployment uses [Ansible](https://www.ansible.com/).

## Contributing

Please feel free to open issues or pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
