# LaTeX Class for the Association for Computing Machinery

**Version:** 2.15 (2025/08/21)  
**Author:** Boris Veytsman

This package provides a class for typesetting publications of the Association for Computing Machinery.

---

## 1. Introduction

The Association for Computing Machinery (ACM) is the world's largest educational and scientific computing society. This consolidated template package replaces all previous independent class files and provides a single up-to-date LaTeX package with optional calls.

**Important Notes:**
- The package uses only free TeX packages and fonts included in TeXLive, MikTeX and other popular TeX distributions
- Uses libertine font set (must be installed before use)
- Fonts cannot be substituted
- Margin adjustments are not allowed

**Development version:** https://github.com/borisveytsman/acmart

---

## 2. User's Guide

### 2.1 Installation

Most users already have this package installed. If not, upgrade your TeX distribution.

**Latest versions:**
- Released: https://www.ctan.org/pkg/acmart
- Development: https://github.com/borisveytsman/acmart

**Manual Installation Steps:**
1. Run `latex acmart.ins` (produces `acmart.cls`)
2. Place `acmart.cls`, `acm-jdslogo.png`, and `ACM-Reference-Format.bst` where LaTeX can find them
3. Update the database of file names
4. Use `acmart.pdf` for documentation

**Required Packages:**
- amsart, babel, balance, booktabs, caption, cmap, comment, draftwatermark
- environ, etoolbox, fancyhdr, float, fontenc, framed, geometry, graphicx
- hyperref, hyperxmp, iftex, libertine, manyfoot, microtype, natbib, newtxmath
- pbalance, refcount, setspace, totpages, unicode-math, xcolor, xkeyval
- xstring, zi4, zref-savepos, zref-user

---

### 2.2 Invocation and Options

**Basic Usage:**
```latex
\documentclass[<options>]{acmart}
```

**Format Options:**

| Format | Usage |
|--------|-------|
| `manuscript` | Manuscript (default) |
| `acmsmall` | Small single-column (journals: JACM, TALLIP, TOCHI, etc.) |
| `acmlarge` | Large single-column (journals: DLT, DTRAP, HEALTH, etc.) |
| `acmtog` | Large double-column (TOG, conference Technical Papers) |
| `sigconf` | Most ACM conferences and all ICPS volumes |
| `sigplan` | SIGPLAN conferences |
| `acmengage` | ACM EngageCSEdu course materials |
| `acmcp` | ACM cover page |

**Example:**
```latex
\documentclass[format=acmsmall, screen=true, review=false]{acmart}
% Or omit 'format=':
\documentclass[acmtog, review=false]{acmart}
```

**Boolean Options:**

| Option | Default | Meaning |
|--------|---------|---------|
| `review` | false | Review version (numbered lines, colored hyperlinks) |
| `screen` | varies | Screen version (colored hyperlinks) |
| `natbib` | true | Use natbib package |
| `anonymous` | false | Anonymous author information |
| `authorversion` | false | Special version for authors' personal use |
| `nonacm` | false | Non-ACM document (no ACM headers/footers) |
| `timestamp` | false | Time stamp in footer |
| `authordraft` | false | Author's draft mode |
| `acmthm` | true | Define theorem-like environments |
| `balance` | true | Balance last page in two-column mode |
| `pbalance` | false | Balance using pbalance package |
| `urlbreakonhyphens` | true | Break URLs on hyphens |

**Notes:**
- Conference proceedings as special issues use journal format
- SIGCHI formats (sigchi, sigchi-a) are retired; use sigconf instead
- `screen` default is false except for PACM (electronic-only)

---

### 2.3 Top Matter

Commands should be used **before** `\maketitle`.

#### Journal/Transaction Name
```latex
\acmJournal{TOMS}
```

#### Conference Information
```latex
\acmConference[<short name>]{<name>}{<date>}{<venue>}

% Examples:
\acmConference[TD'15]{Technical Data Conference}{November 12--16}{Dallas, TX, USA}
\acmConference{SA'15 Art Papers}{November 02--06, 2015}{Kobe, Japan}
```

#### Book Title (optional override)
```latex
\acmBooktitle{Companion to Programming '17}
```

#### Editors
```latex
\editor{Jennifer B. Sartor}
\editor{Theo D'Hondt}
```

#### Title and Subtitle
```latex
\title[<ShortTitle>]{<FullTitle>}
\subtitle{<subtitle>}
```

#### Authors and Affiliations

**Basic Structure:**
```latex
\author{Author Name}
\orcid{ORCID-ID}
\affiliation{%
  \institution{University Name}
  \department{Department Name}
  \city{City}
  \state{State}
  \country{Country}}
\email{email@address}
```

**Important Rules:**
- DO NOT use `\and` or commas between authors
- Each author gets their own `\author` command
- `\institution`, `\city`, and `\country` are mandatory
- ACM strongly encourages ORCID for all authors
- Get free ORCID at http://www.orcid.org/

**Multiple Authors, Same Affiliation:**
```latex
\author{Author One}
\email{one@university.edu}
\author{Author Two}
\email{two@university.edu}
\affiliation{%
  \institution{University Name}
  \city{City}
  \country{Country}}
```

**Multiple Affiliations:**
```latex
\author{Author Name}
\orcid{...}
\affiliation{%
  \institution{First Institution}
  \city{City}
  \country{Country}}
\affiliation{%
  \institution{Second Institution}
  \city{City}
  \country{Country}}
\email{email@address}
```

**Additional Affiliation (use sparingly):**
```latex
\author{Ben Trovato}
\additionalaffiliation{%
  \institution{The Thørväld Group}
  \city{Hekla}
  \country{Iceland}}
\affiliation{%
  \institution{Institute for Clarity}
  \city{Dublin}
  \country{USA}}
```

**Affiliation Elements:**
- `\position` - Position/title
- `\institution` - Institution name (mandatory)
- `\department[<level>]` - Department name
- `\city` - City (mandatory)
- `\state` - State/province
- `\country` - Country (mandatory)

**Note:** `\streetaddress` and `\postcode` are deprecated (produce warnings)

**Multiple Departments:**
```latex
% Independent departments:
\department{Department of Lunar Studies}
\department{John Doe Institute}

% Nested departments (higher numbers = higher in org chart):
\department[0]{Department of Lunar Studies}
\department[1]{John Doe Institute}
```

#### Notes
```latex
\titlenote{This is a title note}
\subtitlenote{This is a subtitle note}
\authornote{This is an author note}
\authornotemark[1]  % Reference to previous note
```

**Never use `\footnote` inside `\author` or `\title`!**

#### Authors' Addresses Override
```latex
\authorsaddresses{%
  Authors' addresses: G.~Zhou, Computer Science Department...}

% Suppress addresses:
\authorsaddresses{}
```

#### Volume, Issue, Article Information
```latex
\acmVolume{9}
\acmNumber{4}
\acmArticle{39}
\acmYear{2010}
\acmMonth{3}  % Numerical
\acmArticleSeq{5}  % Override sequence number
```

#### Submission and Publication IDs
```latex
\acmSubmissionID{123-A56-BU3}
\acmISBN{978-1-4503-3916-2}
\acmDOI{10.1145/9999997.9999999}
```

#### Badges
```latex
\acmBadge[<url>]{<graphics>}
% Example:
\acmBadge[http://ctuning.org/ae/ppopp2016.html]{ae-logo}
```

#### Start Page
```latex
\startPage{42}
```

#### Keywords
```latex
\keywords{wireless sensor networks, media access control, 
  multi-channel, radio interference}
```

#### CCS Concepts
Generate codes at http://dl.acm.org/ccs.cfm, then paste:

```latex
\begin{CCSXML}
<ccs2012>
  <concept>
    <concept_id>10010520.10010553.10010562</concept_id>
    <concept_desc>Computer systems organization~Embedded systems</concept_desc>
    <concept_significance>500</concept_significance>
  </concept>
</ccs2012>
\end{CCSXML}

\ccsdesc[500]{Computer systems organization~Embedded systems}
\ccsdesc[300]{Computer systems organization~Redundancy}
\ccsdesc{Computer systems organization~Robotics}
```

**Required:** For all articles over 2 pages; optional for 1-2 page articles

#### Copyright
```latex
\setcopyright{<status>}
```

**Copyright Status Values:**

| Value | Meaning |
|-------|---------|
| `none` | No copyright/permission information |
| `acmlicensed` | Authors retain copyright, license to ACM |
| `rightsretained` | Authors retain copyright and publication rights |
| `usgov` | All authors are US government employees |
| `usgovmixed` | Some authors are US government employees |
| `cagov` | All authors are Canadian government employees |
| `cagovmixed` | Some authors are Canadian government employees |
| `licensedusgovmixed` | Some US gov employees, licensed to ACM |
| `licensedcagov` | All Canadian gov employees, licensed to ACM |
| `licensedcagovmixed` | Some Canadian gov employees, licensed to ACM |
| `othergov` | Government employees (not US/Canada) |
| `licensedothergov` | Other gov employees, licensed to ACM |
| `iw3c2w3` | IW3C2 conferences |
| `iw3c2w3g` | IW3C2 with Google employees |
| `cc` | Creative Commons license |
| `acmcopyright` | Copyright transferred to ACM (deprecated for non-commissioned) |

**Creative Commons License Type:**
```latex
\setcctype[<version>]{<type>}
% version: 3.0 or 4.0 (default: 4.0)
% type: zero, by, by-sa, by-nd, by-nc, by-nc-sa, by-nc-nd
```

**Copyright Year:**
```latex
\copyrightyear{2015}
```

**Author Version:**
```latex
\documentclass[authorversion=true]{acmart}
```

#### Abstract and Teaser Figure

**Must come before `\maketitle`:**

```latex
\begin{abstract}
This is the abstract text...
\end{abstract}

\begin{teaserfigure}
  \includegraphics[width=\textwidth]{sampleteaser}
  \caption{This is a teaser}
  \label{fig:teaser}
\end{teaserfigure}
```

#### Top Matter Settings
```latex
\settopmatter{<settings>}
```

**Settings:**

| Parameter | Values | Meaning |
|-----------|--------|---------|
| `printccs` | true/false | Print CCS categories |
| `printacmref` | true/false | Print ACM bibliographic entry |
| `printfolios` | true/false | Print page numbers |
| `authorsperrow` | numeric | Authors per row in conference format |

**Example:**
```latex
\settopmatter{printacmref=false, printccs=true, printfolios=true}
```

#### Publication History
```latex
\received{20 February 2007}
\received[revised]{12 March 2009}
\received[accepted]{5 June 2009}
```

#### Generate Title
```latex
\maketitle
```

#### Short Authors (optional override)
```latex
\maketitle
\renewcommand{\shortauthors}{Zhou et al.}
```

---

### 2.4 Top Matter of ACM Engage Materials

ACM Engage uses Creative Commons license (default CC-BY). Override with:
```latex
\setcctype{by-nc}
```

**Engage Metadata:**
```latex
\setengagemetadata{Course}{CS1}
\setengagemetadata{Programming Language}{Python}
\setengagemetadata{Knowledge Unit}{Programming Concepts}
\setengagemetadata{CS Topics}{Functions, Data Types, Expressions}
```

Abstract is called "synopsis" in Engage materials.

---

### 2.5 ACM Cover Page

Format: `acmcp`

**Article Types:**
```latex
\acmArticleType{<type>}
% Types: Research (default), Review, Discussion, Invited, Position
```

**Links:**
```latex
\acmCodeLink{https://github.com/repository/code}
\acmDataLink{https://datadryad.org/stash/dataset/doi:DOI}
```

**Contributions:**
```latex
\acmContributions{AW designed the study, CD performed it...}
```

**Required Sections:**
- Problem statement
- Methods
- Results
- Significance

---

### 2.6 Internationalization

**Set Languages:**
```latex
\documentclass[sigconf, language=french, language=english]{acmart}
```

Last language is main language; others are secondary.

**Translated Content:**
```latex
\title{A note on computational complexity}
\translatedtitle{french}{Remarque sur la complexité de calcul}

\keywords{main language keywords}
\translatedkeywords{french}{mots-clés en français}

\begin{abstract}
Main language abstract
\end{abstract}

\begin{translatedabstract}{french}
Résumé en français
\end{translatedabstract}
```

---

### 2.7 Algorithms

Use standard algorithm packages:
- algorithm2e
- algorithms
- listings

Authors free to choose preferred package.

---

### 2.8 Figures and Tables

**Important Rules:**
1. Figure captions go **after** figure bodies
2. Table captions go **before** table bodies

**Figure Types:**

| Type | Usage |
|------|-------|
| `figure` | Standard (full width in 1-col; column width in 2-col) |
| `figure*` | Full text width in 2-column format |
| `table` | Standard table |
| `table*` | Full width table in 2-column format |
| `teaserfigure` | Before `\maketitle` |

**Image Scaling:**
- Teaser, 1-column figure, or 2-column `figure*`: use `\textwidth`
- 2-column `figure`: use `\columnwidth`

**Table Guidelines (booktabs package):**
1. Never use vertical rules
2. Never use double rules
3. Don't overuse horizontal rules

**Table Footnotes:**
```latex
\begin{table}
\caption{Simulation Configuration}
\begin{minipage}{\columnwidth}
\begin{center}
\begin{tabular}{ll}
\toprule
TERRAIN\footnote{Table footnote text} & Value\\
\midrule
Node Number & 289\\
\bottomrule
\end{tabular}
\end{center}
\bigskip
\footnotesize\emph{Source:} Table source note.
\emph{Note:} Additional table note.
\end{minipage}
\end{table}
```

---

### 2.9 Descriptions of Images

**Required for Accessibility:**
```latex
\begin{figure}
  \centering
  \includegraphics{voltage}
  \Description{A bell-like histogram centered at 0.5V with 
    most measurements between 0.2V and 0.8V}
  \caption{Histogram of the measurements of voltage}
  \label{fig:voltage}
\end{figure}
```

---

### 2.10 Theorems

**Predefined Environments:**
- Style `acmplain`: theorem, conjecture, proposition, lemma, corollary
- Style `acmdefinition`: example, definition

**Disable with:** `acmthm=false`

**Custom Theorem Environments:**
```latex
\AtEndPreamble{%
  \theoremstyle{acmdefinition}
  \newtheorem{remark}[theorem]{Remark}}
```

**With cleveref and Shared Counters:**
```latex
\newtheorem{theorem}{Theorem}[section]
\newtheorem{lemma}[theorem]{Lemma}

\begin{theorem}\label{thm:test}
...
\begin{lemma}\label[lemma]{test-lemma}
```

---

### 2.11 Online-only and Offline-only Material

```latex
\begin{printonly}
Supplementary materials are available online.
\end{printonly}

\begin{screenonly}
(Actual supplementary materials here)
\end{screenonly}
```

**Requirements:**
- `\begin` and `\end` must start on their own lines
- No leading or trailing spaces

---

### 2.12 Anonymous Mode

**Block Suppression:**
```latex
\begin{anonsuppress}
This is a continuation of previous work by the author \cite{prev1, prev2}.
\end{anonsuppress}
```

**Inline Suppression:**
```latex
This work was performed at \anon{NSA}.
This work was performed at \anon[No Such Agency]{NSA}.
```

---

### 2.13 Acknowledgments

```latex
\begin{acks}
The authors thank Dr. Smith for assistance.

This work is supported by the 
\grantsponsor{GS501100001809}{National Natural Science Foundation of China}
{https://doi.org/10.13039/501100001809} under Grant 
No.:~\grantnum{GS501100001809}{61273304} 
and~\grantnum[http://www.nnsf.cn/youngscientists]{GS501100001809}
{Young Scientists' Support Program}.
\end{acks}
```

**Grant Commands:**
```latex
\grantsponsor{<sponsorID>}{<name>}{<url>}
\grantnum[<url>]{<sponsorID>}{<number>}
```

**Notes:**
- Automatically omitted in anonymous mode
- `\begin{acks}` and `\end{acks}` on their own lines

---

### 2.14 Bibliography

#### 2.14.1 Processing using BibTeX

**Default Style:** `ACM-Reference-Format.bst`

**Citation Styles:**
```latex
\citestyle{acmauthoryear}  % Author-year
\citestyle{acmnumeric}     % Numeric (default for most)
```

**Customize natbib:**
```latex
\setcitestyle{numbers,sort&compress}
\setcitestyle{nosort}
```

**Citation Commands:**
- `\cite`, `\citep`, `\citeauthor` (natbib)
- `\shortcite` - year only in author-date; same as `\cite` in numeric
- `\citeyear` - year only (no parentheses)
- `\citeyearpar` - year in parentheses

#### 2.14.2 URL, DOI, eprint

**BibTeX Fields:**
```bibtex
doi = "10.1145/1188913.1188915",
url = "http://example.com/paper.pdf",
distinctURL = 1,  % Print URL even if DOI present
lastaccessed = {March 2, 2005},
archived = {https://web.archive.org/web/20240505055615},
eprint = "960935712",
primaryclass = "cs",
```

#### 2.14.3 Special Entry Types

**Online/Web Pages:**
```bibtex
@online{Thornburg01,
  author = {Harry Thornburg},
  year = {2001},
  title = {Introduction to Bayesian Statistics},
  url = {http://example.com/bayes.html},
  month = mar,
  lastaccessed = {March 2, 2005},
}
```

**Software and Datasets:**
```bibtex
@ArtifactSoftware{R,
  title = {R: A Language and Environment for Statistical Computing},
  author = {{R Core Team}},
  organization = {R Foundation for Statistical Computing},
  year = {2019},
  url = {https://www.R-project.org/},
}

@ArtifactDataset{UMassCitations,
  author = {Sam Anzaroot and Andrew McCallum},
  title = {{UMass} Citation Field Extraction Dataset},
  year = 2013,
  url = {http://www.iesl.cs.umass.edu/data/...},
  lastaccessed = {May 27, 2019}
}
```

**Video/Audio (standalone):**
```bibtex
@online{Obama08,
  author = {Barack Obama},
  year = {2008},
  title = {A more perfect union},
  howpublished = {Video},
  url = {http://video.google.com/...},
  lastaccessed = {March 21, 2008},
}
```

**Video/Audio (in proceedings):**
```bibtex
@Inproceedings{Novak03,
  author = {Dave Novak},
  title = {Solder man},
  booktitle = {ACM SIGGRAPH 2003 Video Review},
  howpublished = {Video},
  doi = {10.9999/woot07-S422},
}
```

**Complete Journal Issue:**
```bibtex
@periodical{JCohen96,
  key = {Cohen},
  editor = {Jacques Cohen},
  title = {Special issue: Digital Libraries},
  journal = {Communications of the {ACM}},
  volume = {39},
  number = {11},
  year = {1996},
}
```

**Presentation (unpublished):**
```bibtex
@Presentation{Reiser2014,
  author = {Brian J. Reiser},
  year = 2014,
  title = {Designing coherent storylines aligned with NGSS},
  venue = {National Science Education Leadership Association Meeting, 
           Boston, MA, USA},
  url = {https://www.academia.edu/6884962/}
}
```

**Preprint:**
```bibtex
@preprint{AnzarootPBM14,
  author = {Sam Anzaroot and others},
  title = {Learning Soft Linear Constraints},
  year = {2014},
  archivePrefix = {arXiv},
  eprint = {1403.1349},
  doi = {10.48550/arXiv.1403.1349}
}
```

**Under Review:**
```bibtex
@underreview{Baggett2025,
  author = {R. Baggett and others},
  year = 2025,
  title = {Fluidity in the Phased Framework...},
  journal = {Robotics Aut. Systems}
}
```

#### 2.14.4 Dates and Sorting

**No Date:** Style adds "[n. d.]" automatically

**No Author (use key for sorting):**
```bibtex
@online{TUGInstmem,
  key = {TUG},
  year = 2017,
  title = {Institutional members of {\TeX} Users Group},
  url = {http://tug.org/instmem.html},
}
```

**Sorting Names with "van/von":**
```bibtex
@PREAMBLE{"\providecommand{\noopsort}[1]{}"}

@article{...,
  author = {Ludwig {\noopsort{Beethoven}}van Beethoven},
}
```

**Journal Name Strings:** All ACM journals have predefined strings (e.g., `journal = taccess`)

#### 2.14.5 Processing using BibLaTeX

**Setup:**
```latex
\documentclass[natbib=false]{acmart}

\RequirePackage[
  datamodel=acmdatamodel,
  style=acmnumeric,  % or style=acmauthoryear
]{biblatex}

\addbibresource{software.bib}
\addbibresource{sample-base.bib}
```

**At end of document:**
```latex
\printbibliography
```

**See samples:** `sample-*-biblatex.tex`

---

### 2.15 Colors

**Predefined ACM Colors:**
- ACMBlue
- ACMYellow
- ACMOrange
- ACMRed
- ACMLightBlue
- ACMGreen
- ACMPurple
- ACMDarkBlue

**Accessibility Recommendations:**
1. Ensure readability in greyscale
2. Consider color vision deficiencies (8% males, 0.5% females)
3. Most printing is black and white
4. Don't encode information using only color

**Color Selection Tools:**
- ColourBrewer: http://colorbrewer2.org/
- ACE: http://daprlab.com/ace/

---

### 2.15.1 Manual Bibliography

**Not recommended!** ACM submissions need metadata extraction. Manual bibliographies without special macros slow publication.

Use BibTeX or BibLaTeX with ACM styles.

---

### 2.16 Other Notable Packages

**Recommended:**
- `subcaption` - subfigures with separate captions
- `nomencl` - list of symbols
- `glossaries` - list of concepts

**Typography Penalties:**
```latex
\widowpenalty=10000   % Prevent widows
\clubpenalty=10000    % Prevent orphans
\brokenpenalty=10000  % Prevent hyphen at page end
```

Reduce these for strict page limits (but results may be less optimal).

**Line Breaking:**
- Use `\sloppy` or `sloppypar` environment for problematic paragraphs
- Use `\NoCaseChange` in titles to prevent extraneous uppercasing

---

### 2.17 Counting Words

**For papers with word count limits:**

Add to beginning of document:
```latex
%TC:macro \cite [option:text,text]
%TC:macro \citep [option:text,text]
%TC:macro \citet [option:text,text]
%TC:envir table 0 1
%TC:envir table* 0 1
%TC:envir tabular [ignore] word
%TC:envir displaymath 0 word
%TC:envir math 0 word
%TC:envir comment 0 0
```

Use `\begin{math}...\end{math}` instead of `$...$`.

Run: `texcount yourfile.tex` for word count report.

**Note:** Count is approximate; final decision based on PDF count.

---

### 2.18 Disabled or Forbidden Commands

**Restrictions:**
- Cannot put multiple authors or emails in one `\author` or `\email` command
- Cannot change `\baselinestretch` (produces error)
- Should not abuse `\vspace` (disturbs typesetting)
- Should not load `amssymb` package (acmart defines symbols itself)

---

### 2.19 Notes for Wizards

**Hook File:** `acmart-preload-hook.tex`

Use to load packages before acmart or pass options to packages loaded by acmart.

**Example - Load titletoc before hyperref:**
```latex
\let\LoadClassOrig\LoadClass
\renewcommand\LoadClass[2][]{\LoadClassOrig[#1]{#2}%
  \usepackage{titletoc}}
```

**Example - Pass options to xcolor:**
```latex
\PassOptionsToPackage{dvipsnames}{xcolor}
```

**Warning:** Using this hook can create unacceptable or non-compiling manuscripts. Use at your own risk!

**Another Hook:**
```latex
\AtBeginMaketitle{...}  % Executed before \maketitle
```

---

### 2.20 Currently Supported Publications

**ACM Journals and Transactions:**

| Abbreviation | Publication |
|--------------|-------------|
| ACMJCSS | ACM Journal on Computing and Sustainable Societies |
| ACMJDS | ACM Journal of Data Science |
| AILET | ACM AI Letters |
| CIE | ACM Computers in Entertainment |
| CSUR | ACM Computing Surveys |
| DLT | Distributed Ledger Technologies: Research and Practice |
| DGOV | Digital Government: Research and Practice |
| DTRAP | Digital Threats: Research and Practice |
| FAC | Formal Aspects of Computing |
| GAMES | ACM Games: Research and Practice |
| HEALTH | ACM Transactions on Computing for Healthcare |
| IMWUT | PACM on Interactive, Mobile, Wearable and Ubiquitous Technologies |
| JACM | Journal of the ACM |
| JATS | ACM Journal on Autonomous Transportation Systems |
| JDIQ | ACM Journal of Data and Information Quality |
| JDS | ACM/IMS Journal of Data Science |
| JEA | ACM Journal of Experimental Algorithmics |
| JERIC | ACM Journal of Educational Resources in Computing |
| JETC | ACM Journal on Emerging Technologies in Computing Systems |
| JOCCH | ACM Journal on Computing and Cultural Heritage |
| JRC | ACM Journal on Responsible Computing |
| PACMCGIT | Proceedings of the ACM on Computer Graphics and Interactive Techniques |
| PACMHCI | PACM on Human-Computer Interaction |
| PACMOD | PACM on Management of Data |
| PACMNET | PACM on Networking |
| PACMPL | PACM on Programming Languages |
| PACMSE | PACM on Software Engineering |
| POMACS | PACM on Measurement and Analysis of Computing Systems |
| TAAS | ACM Transactions on Autonomous and Adaptive Systems |
| TACCESS | ACM Transactions on Accessible Computing |
| TACO | ACM Transactions on Architecture and Code Optimization |
| TAIS | ACM Transactions on AI for Science |
| TAISAP | ACM Transactions on AI Security and Privacy |
| TALG | ACM Transactions on Algorithms |
| TALLIP | ACM Transactions on Asian and Low-Resource Language Information Processing |
| TAP | ACM Transactions on Applied Perception |
| TCPS | ACM Transactions on Cyber-Physical Systems |
| TDS | ACM/IMS Transactions on Data Science |
| TEAC | ACM Transactions on Economics and Computation |
| TECS | ACM Transactions on Embedded Computing Systems |
| TELO | ACM Transactions on Evolutionary Learning and Optimization |
| THRI | ACM Transactions on Human-Robot Interaction |
| TIIS | ACM Transactions on Interactive Intelligent Systems |
| TIOT | ACM Transactions on Internet of Things |
| TISSEC | ACM Transactions on Information and System Security |
| TIST | ACM Transactions on Intelligent Systems and Technology |
| TKDD | ACM Transactions on Knowledge Discovery from Data |
| TMIS | ACM Transactions on Management Information Systems |
| TOCE | ACM Transactions on Computing Education |
| TOCHI | ACM Transactions on Computer-Human Interaction |
| TOCL | ACM Transactions on Computational Logic |
| TOCS | ACM Transactions on Computer Systems |
| TOCT | ACM Transactions on Computation Theory |
| TODAES | ACM Transactions on Design Automation of Electronic Systems |
| TODS | ACM Transactions on Database Systems |
| TOG | ACM Transactions on Graphics |
| TOIS | ACM Transactions on Information Systems |
| TOIT | ACM Transactions on Internet Technology |
| TOMACS | ACM Transactions on Modeling and Computer Simulation |
| TOMM | ACM Transactions on Multimedia Computing, Communications and Applications |
| TOMPECS | ACM Transactions on Modeling and Performance Evaluation of Computing Systems |
| TOMS | ACM Transactions on Mathematical Software |
| TOPC | ACM Transactions on Parallel Computing |
| TOPLAS | ACM Transactions on Programming Languages and Systems |
| TOPML | Transactions on Probabilistic Machine Learning |
| TOPS | ACM Transactions on Privacy and Security |
| TORS | ACM Transactions on Recommender Systems |
| TOS | ACM Transactions on Storage |
| TOSEM | ACM Transactions on Software Engineering and Methodology |
| TOSN | ACM Transactions on Sensor Networks |
| TQC | ACM Transactions on Quantum Computing |
| TRETS | ACM Transactions on Reconfigurable Technology and Systems |
| TSAS | ACM Transactions on Spatial Algorithms and Systems |
| TSC | ACM Transactions on Social Computing |
| TSLP | ACM Transactions on Speech and Language Processing |
| TWEB | ACM Transactions on the Web |

**Special:** FACMP - forthcoming ACM publication (for journals not yet assigned ISSN)

---

### 2.21 Samples

**Available Templates:**

- `sample-manuscript` - Proceedings paper in manuscript format
- `sample-acmsmall` - Journal paper in acmsmall format
- `sample-acmsmall-biblatex` - Journal with BibLaTeX
- `sample-acmlarge` - Journal paper in acmlarge format
- `sample-acmtog` - Journal paper in acmtog format
- `sample-sigconf` - Standard conference proceedings (sigconf)
- `sample-sigconf-biblatex` - Conference with BibLaTeX
- `sample-sigconf-authordraft` - Conference with authordraft option
- `sample-sigconf-i13n` - Conference with multilanguage titles/abstract
- `sample-sigconf-xelatex` - Conference with XeLaTeX
- `sample-sigconf-lualatex` - Conference with LuaLaTeX
- `sample-sigplan` - SIGPLAN conference proceedings
- `sample-acmsmall-conf` - Conference published in journal (acmsmall)
- `sample-acmtog-conf` - Conference published in journal (acmtog)
- `sample-acmcp` - ACM Cover Page (JDS)
- `sample-acmengage` - ACM Engage publication

---

### 2.22 SIGCHI Extended Abstract Format (sigchi-a)

**RETIRED as of Spring 2020!**

ACM will NOT accept documents in this format for publication.

**Only for non-ACM use:**
```latex
\documentclass[sigchi-a, nonacm]{acmart}
```

**Special Margin Environments:**
- `sidebar` - Textual information in margin
- `marginfigure` - Figure in margin
- `margintable` - Table in margin

**Figure Widths:**
- `figure`: `\columnwidth`
- `marginfigure`: `\marginparwidth`
- `figure*`: `\fulltextwidth`

---

### 2.23 Experiments with Tagging

**Experimental PDF/A Accessibility:**

ACM is working on fully tagged PDFs compliant with accessibility standards.

**To Try (experimental, unsupported):**
1. Use `\documentclass{acmart-tagged}`
2. Add `\DocumentMetadata[<options>]` in preamble

See: `sample-acmsmall-tagged.tex`

**Resources:**
- https://www.latex-project.org/publications/indexbytopic/pdf/
- https://tug.org/twg/accessibility/overview.html

**Bug reports welcome:** https://github.com/borisveytsman/acmart/issues

---

## Quick Reference Summary

### Essential Document Structure

```latex
\documentclass[sigconf]{acmart}  % or acmsmall, acmtog, etc.

% Copyright
\setcopyright{acmlicensed}
\copyrightyear{2025}

% Conference/Journal
\acmConference[CONF'25]{Conference Name}{Date}{Location}
% OR
\acmJournal{JACM}

% Volume/Issue (journals)
\acmYear{2025}
\acmVolume{1}
\acmNumber{1}
\acmArticle{1}

% DOI
\acmDOI{10.1145/1234567.1234568}

% Title
\title{Your Title Here}

% Authors
\author{First Author}
\orcid{0000-0000-0000-0000}
\affiliation{%
  \institution{University Name}
  \city{City}
  \country{Country}}
\email{first@university.edu}

\author{Second Author}
\affiliation{%
  \institution{Another University}
  \city{City}
  \country{Country}}
\email{second@another.edu}

% CCS Concepts
\begin{CCSXML}
<!-- Paste from http://dl.acm.org/ccs.cfm -->
\end{CCSXML}
\ccsdesc[500]{Computer systems organization~Embedded systems}

% Keywords
\keywords{keyword1, keyword2, keyword3}

% Abstract
\begin{abstract}
Your abstract here.
\end{abstract}

\maketitle

% Body
\section{Introduction}
Your content...

% Acknowledgments
\begin{acks}
Thanks to...
\end{acks}

% Bibliography
\bibliographystyle{ACM-Reference-Format}
\bibliography{yourbib}

\end{document}
```

---

## Common Pitfalls to Avoid

1. **Don't** put abstract after `\maketitle` (error)
2. **Don't** use `\and` between authors
3. **Don't** put multiple authors in one `\author` command
4. **Don't** forget mandatory fields: `\institution`, `\city`, `\country`
5. **Don't** use `\footnote` inside `\author` or `\title`
6. **Do** put figure captions after figure, table captions before table
7. **Do** include `\Description` for all figures (accessibility)
8. **Do** use `\textwidth` for 1-column, `\columnwidth` for 2-column figures
9. **Do** get CCS concepts from http://dl.acm.org/ccs.cfm
10. **Do** include ORCID for all authors

---

## Getting Help

- GitHub Issues: https://github.com/borisveytsman/acmart/issues
- TeX Stack Exchange: https://tex.stackexchange.com
- TeX Users Group: https://tug.org
- ACM Formatting: https://www.acm.org/publications/authors/
