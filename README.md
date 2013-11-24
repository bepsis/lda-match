NB: this is now deprecated in favor of the similarly named [ldamatch](https://github.com/cslu-nlp/ldamatch), written in pure R and introducing additional features.

Introduction
============

`lda-match2.py` and `lda-match3.py` are Python 2 and Python 3 versions 
(respectively) of scripts which allow the user to generate a subsample of 
data (represented by a comma-separated values file) in which two groups do
not differ (according to a two-tailed unequal variance _t_-test) on an 
arbitrary number of real-valued measures.

The approach here is similar to the "greedy" algorithm used by van Santen
et al. (2010; _Autism_) but in general results in larger subgroups, as it
uses linear discriminant analysis (LDA) to identify outliers. To a first
approximation, when the _t_-test assumptions (normality or large samples),
the assumptions of LDA will also hold, and therefore this will give
something close to an "optimal" subsample according to a criterion which 
favors large subsamples of approximately the same size.

Installation instructions
=========================

This code requires Python (either 2 or 3) and two additional packages, 
`numpy` and `rpy2`. You will need all the following:

* Python (try `python --version`) 
* R (try `R --version`)
* C compiler (try `cc --version`)
* `pip`, the Python package manager (try `pip`)
* `numpy` and `rpy2`:

    `sudo pip install numpy rpy2`

Usage
=====

Users _must_ specify the labels for the two groups (`-a`, `-b`), the 
column name containing the groups (`-g`), the column name(s) of the 
feature(s) to match on (-m), and the location of the input file. Users 
_may_ specify that observations are only to be removed from the first 
group (`-d`), the two-tailed alpha level (`-p`, by default .2), and the location for the output file (`-o`; by default, results are printed to 
`STDOUT`).

For more information, refer to the worked examples below.

Worked examples
===============

* Match ALN and ALI children in `DX.csv` on chronological age (`CA`), ADOS severity score (`ADOS`), and SCQ total score (`SCQ`) at two-tailed alpha >= 0.2, and write the resulting set to a file called `TD-SLI.csv`:

    `python lda-match2.py -a TD -b SLI -g DX -m CA -m ADOS -m SCQ -p 0.2 -o TD-SLI.csv DX.csv`

* Match TD and ALN/ALI children in `DX.csv` on chronological age (`CA`) and non-verbal IQ (`NVIQ`) at two-tailed alpha >= .5 and write the resulting set to a file called `TD-ASD-p5.csv`:

    `python lda-match2.py -a TD -b ALN -b ALI -g DX -m CA -m NVIQ -p 0.5 -o TD-ASD-p5.csv DX.csv`

* Alternative conventions for generating the previous set:
    
    `python lda-match2.py -aTD -bALN,ALI -gDX -mCA,NVIQ -p.05 DX.csv > TD-ASD-p5.csv`

Some results on the ERPA data
=============================

* TD (36) vs. SLI (20): CA! [+1]
* ALN (25) vs. TD (28): CA, NVIQ, VIQ! [+1]
* ALN (23) vs. ALI (24): CA, ADOS!, SCQ [+2]
* TD (42) vs. ASD (ALN = 24, ALI = 19): CA, NVIQ! [0]
* LN (ALN = 26, TD = 39) vs. LI (ALI = 26, SLI = 20): CA! [+2]
* ALI (26) vs. SLI (20): CA, NVIQ, VIQ (no matching necessary) [+3]
* ASD (ALN = 25, ALI = 26) vs. nASD (TD = 44, SLI = 20): CA, NVIQ (no matching necessary) [+6]

Key:

* !: last feature matched
* \[+N\]: change in overall subsample size compared to the "greedy" method
operating at the same alpha level

License
=======

BSD-like (see the source)

Author
======

Kyle Gorman (<gormanky@ohsu.edu>), with thanks to Steven Bedrick
