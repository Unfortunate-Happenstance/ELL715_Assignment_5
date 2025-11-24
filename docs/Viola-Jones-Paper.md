# Rapid Object Detection using a Boosted Cascade of Simple Features

**Authors:**  
- Paul Viola (viola@merl.com) - Mitsubishi Electric Research Labs, 201 Broadway, 8th FL, Cambridge, MA 02139
- Michael Jones (mjones@crl.dec.com) - Compaq CRL, One Cambridge Center, Cambridge, MA 02142

**Conference:** ACCEPTED CONFERENCE ON COMPUTER VISION AND PATTERN RECOGNITION 2001

---

## Abstract

This paper describes a machine learning approach for visual object detection which is capable of processing images extremely rapidly and achieving high detection rates. This work is distinguished by three key contributions:

1. **The Integral Image**: A new image representation called the "Integral Image" which allows the features used by our detector to be computed very quickly.

2. **AdaBoost Learning Algorithm**: A learning algorithm, based on AdaBoost, which selects a small number of critical visual features from a larger set and yields extremely efficient classifiers.

3. **Cascade Architecture**: A method for combining increasingly more complex classifiers in a "cascade" which allows background regions of the image to be quickly discarded while spending more computation on promising object-like regions.

The cascade can be viewed as an object specific focus-of-attention mechanism which unlike previous approaches provides statistical guarantees that discarded regions are unlikely to contain the object of interest.

**Performance:** In the domain of face detection the system yields detection rates comparable to the best previous systems. Used in real-time applications, the detector runs at 15 frames per second without resorting to image differencing or skin color detection.

---

## 1. Introduction

This paper brings together new algorithms and insights to construct a framework for robust and extremely rapid object detection. This framework is demonstrated on, and in part motivated by, the task of face detection.

### Key Achievement

We have constructed a frontal face detection system which achieves detection and false positive rates which are equivalent to the best published results. This face detection system is most clearly distinguished from previous approaches in its ability to detect faces extremely rapidly.

**Performance Metrics:**
- Operating on 384 by 288 pixel images
- Faces detected at 15 frames per second on a conventional 700 MHz Intel Pentium III
- Works only with information present in a single grey scale image
- Alternative sources of information (image differences, pixel color) can be integrated for even higher frame rates

### Three Main Contributions

#### 1. Integral Image Representation

A new image representation called an integral image that allows for very fast feature evaluation. The system uses a set of features which are reminiscent of Haar Basis functions (though we also use related filters which are more complex than Haar filters).

**Key Properties:**
- The integral image can be computed from an image using a few operations per pixel
- Once computed, any one of these Haar-like features can be computed at any scale or location in constant time

#### 2. Feature Selection using AdaBoost

A method for constructing a classifier by selecting a small number of important features using AdaBoost. Within any image sub-window the total number of Haar-like features is very large, far larger than the number of pixels.

**Learning Process:**
- The weak learner is constrained so that each weak classifier returned can depend on only a single feature
- Each stage of the boosting process, which selects a new weak classifier, can be viewed as a feature selection process
- AdaBoost provides an effective learning algorithm and strong bounds on generalization performance

#### 3. Cascade Structure

A method for combining successively more complex classifiers in a cascade structure which dramatically increases the speed of the detector by focusing attention on promising regions of the image.

**Key Insight:** It is often possible to rapidly determine where in an image an object might occur. More complex processing is reserved only for these promising regions.

**Critical Measure:** The "false negative" rate of the attentional process. It must be the case that all, or almost all, object instances are selected by the attentional filter.

**Performance Example:**
- Using a classifier constructed from two Haar-like features: fewer than 1% false negatives and 40% false positives
- Effect: reduces by over one half the number of locations where the final detector must be evaluated

### Cascade Structure Details

Sub-windows which are not rejected by the initial classifier are processed by a sequence of classifiers, each slightly more complex than the last. If any classifier rejects the sub-window, no further processing is performed. The structure of the cascaded detection process is essentially that of a degenerate decision tree.

### Practical Applications

An extremely fast face detector will have broad practical applications:
- User interfaces
- Image databases
- Teleconferencing
- Applications where rapid frame-rates are not necessary (allows significant additional post-processing and analysis)
- Implementation on small low power devices (hand-helds and embedded processors)

**Implementation Example:** This face detector was implemented on the Compaq iPaq handheld and achieved detection at two frames per second (device has a low power 200 mips Strong Arm processor which lacks floating point hardware).

---

## 2. Features

Our object detection procedure classifies images based on the value of simple features.

### Motivations for Using Features

1. **Domain Knowledge Encoding**: Features can act to encode ad-hoc domain knowledge that is difficult to learn using a finite quantity of training data

2. **Speed**: The feature-based system operates much faster than a pixel-based system

### Rectangle Features

The simple features used are reminiscent of Haar basis functions which have been used by Papageorgiou et al. More specifically, we use three kinds of features:

#### Feature Types (See Figure 1 in paper)

1. **Two-Rectangle Feature**: The value is the difference between the sum of the pixels within two rectangular regions. The regions have the same size and shape and are horizontally or vertically adjacent.

2. **Three-Rectangle Feature**: Computes the sum within two outside rectangles subtracted from the sum in a center rectangle.

3. **Four-Rectangle Feature**: Computes the difference between diagonal pairs of rectangles.

**Feature Set Size:** Given that the base resolution of the detector is 24x24, the exhaustive set of rectangle features is quite large, **over 180,000**. Note that unlike the Haar basis, the set of rectangle features is overcomplete.

> **Note on Completeness:** A complete basis has no linear dependence between basis elements and has the same number of elements as the image space, in this case 576. The full set of 180,000 thousand features is many times over-complete.

### 2.1. Integral Image

Rectangle features can be computed very rapidly using an intermediate representation for the image which we call the integral image.

> **Related Work:** There is a close relation to "summed area tables" as used in graphics. We choose a different name here in order to emphasize its use for the analysis of images, rather than for texture mapping.

#### Definition

The integral image at location (x, y) contains the sum of the pixels above and to the left of (x, y), inclusive:

```
ii(x,y) = Σ(x'≤x, y'≤y) i(x',y')
```

where `ii(x,y)` is the integral image and `i(x,y)` is the original image.

#### Computing the Integral Image

Using the following pair of recurrences:

```
s(x,y) = s(x,y-1) + i(x,y)                    (1)
ii(x,y) = ii(x-1,y) + s(x,y)                  (2)
```

where:
- `s(x,y)` is the cumulative row sum
- `s(x,-1) = 0`
- `ii(-1,y) = 0`

The integral image can be computed in **one pass** over the original image.

#### Computing Rectangle Sums (See Figure 2 in paper)

Using the integral image, any rectangular sum can be computed in **four array references**.

**Example:** The sum of the pixels within rectangle D can be computed as:
```
Sum(D) = ii(4) + ii(1) - ii(2) - ii(3)
```

where points 1, 2, 3, 4 are the corners of rectangle D.

**Feature Computation Complexity:**
- The difference between two rectangular sums can be computed in **eight references**
- Two-rectangle features: **six array references**
- Three-rectangle features: **eight array references**
- Four-rectangle features: **nine array references**

### 2.2. Feature Discussion

Rectangle features are somewhat primitive when compared with alternatives such as steerable filters. Steerable filters, and their relatives, are excellent for:
- Detailed analysis of boundaries
- Image compression
- Texture analysis

**Limitations of Rectangle Features:**
- Quite coarse compared to steerable filters
- Only orientations available are vertical, horizontal, and diagonal
- While sensitive to edges, bars, and other simple image structure, they lack flexibility

**Advantages:**
- Provide a rich image representation which supports effective learning
- In conjunction with the integral image, the efficiency provides ample compensation for their limited flexibility

---

## 3. Learning Classification Functions

Given a feature set and a training set of positive and negative images, any number of machine learning approaches could be used to learn a classification function. In our system a variant of AdaBoost is used both to select a small set of features and train the classifier.

### AdaBoost Background

In its original form, the AdaBoost learning algorithm is used to boost the classification performance of a simple (sometimes called weak) learning algorithm.

**Formal Guarantees:**
1. Freund and Schapire proved that the training error of the strong classifier approaches zero exponentially in the number of rounds
2. Generalization performance is related to the margin of the examples
3. AdaBoost achieves large margins rapidly

### Feature Selection Challenge

**Problem:** There are over 180,000 rectangle features associated with each image sub-window, a number far larger than the number of pixels. Even though each feature can be computed very efficiently, computing the complete set is prohibitively expensive.

**Hypothesis:** A very small number of these features can be combined to form an effective classifier (borne out by experiment).

### Weak Learning Algorithm

The weak learning algorithm is designed to select the single rectangle feature which best separates the positive and negative examples.

**Process:**
- For each feature, the weak learner determines the optimal threshold classification function
- Goal: minimum number of examples are misclassified

**Weak Classifier Form:**

A weak classifier h_j(x, f_j, p_j, θ_j) consists of:
- Feature f_j
- Threshold θ_j
- Parity p_j indicating the direction of the inequality sign

```
h_j(x) = {
  1  if p_j·f_j(x) < p_j·θ_j
  0  otherwise
}
```

where x is a 24x24 pixel sub-window of an image.

### Performance Characteristics

In practice no single feature can perform the classification task with low error:
- Features selected in **early rounds**: error rates between 0.1 and 0.3
- Features selected in **later rounds**: error rates between 0.4 and 0.5 (as the task becomes more difficult)

### 3.1. Learning Discussion

Many general feature selection procedures have been proposed. Our final application demanded a very aggressive approach which would discard the vast majority of features.

**Related Work:**
- **Papageorgiou et al.**: Proposed feature selection based on feature variance. Demonstrated good results selecting 37 features out of a total 1734 features.
- **Roth et al.**: Proposed feature selection based on the Winnow exponential perceptron learning rule. The Winnow learning process converges to a solution where many weights are zero. Nevertheless, a very large number of features are retained (perhaps a few hundred or thousand).

### 3.2. Learning Results

Initial experiments demonstrated that a frontal face classifier constructed from **200 features** yields:
- Detection rate: 95%
- False positive rate: 1 in 14,084

**Computation Performance:**
- This classifier requires 0.7 seconds to scan a 384 by 288 pixel image
- Probably faster than any other published system
- Unfortunately, adding features to improve detection performance directly increases computation time

### Selected Features Interpretation

The initial rectangle features selected by AdaBoost are meaningful and easily interpreted (See Figure 3 in paper).

**First Feature:**
- Focuses on the property that the region of the eyes is often darker than the region of the nose and cheeks
- Measures the difference in intensity between the eye region and upper cheeks
- Relatively large in comparison with the detection sub-window
- Should be somewhat insensitive to size and location of the face

**Second Feature:**
- Relies on the property that the eyes are darker than the bridge of the nose
- Compares the intensities in the eye regions to the intensity across the bridge of the nose

---

## 4. The Attentional Cascade

This section describes an algorithm for constructing a cascade of classifiers which achieves increased detection performance while radically reducing computation time.

### Key Insight

Smaller, and therefore more efficient, boosted classifiers can be constructed which reject many of the negative sub-windows while detecting almost all positive instances. The threshold of a boosted classifier can be adjusted so that the false negative rate is close to zero.

**Strategy:** Simpler classifiers are used to reject the majority of sub-windows before more complex classifiers are called upon to achieve low false positive rates.

### Cascade Structure (See Figure 4 in paper)

The overall form of the detection process is that of a degenerate decision tree, what we call a "cascade."

**Process:**
1. A positive result from the first classifier triggers the evaluation of a second classifier (adjusted to achieve very high detection rates)
2. A positive result from the second classifier triggers a third classifier, and so on
3. A negative outcome at any point leads to the immediate rejection of the sub-window

### Stage Construction

Stages in the cascade are constructed by:
1. Training classifiers using AdaBoost
2. Adjusting the threshold to minimize false negatives

**Note:** The default AdaBoost threshold is designed to yield a low error rate on the training data. In general, a lower threshold yields higher detection rates and higher false positive rates.

### Example: First Stage Classifier

An excellent first stage classifier can be constructed from a two-feature strong classifier:
- By reducing the threshold to minimize false negatives
- Against a validation training set: detects 100% of the faces with a false positive rate of 40%
- Computation: about 60 microprocessor instructions
- Much simpler than alternatives (scanning a simple image template or single layer perceptron would require at least 20 times as many operations per sub-window)

### Design Philosophy

The structure of the cascade reflects the fact that within any single image an overwhelming majority of sub-windows are negative.

**Goals:**
- Reject as many negatives as possible at the earliest stage possible
- While a positive instance will trigger the evaluation of every classifier in the cascade, this is an exceedingly rare event

### Training Characteristics

Much like a decision tree, subsequent classifiers are trained using those examples which pass through all the previous stages:
- The second classifier faces a more difficult task than the first
- Examples which make it through the first stage are "harder" than typical examples
- Deeper classifiers face more difficult examples
- This pushes the entire receiver operating characteristic (ROC) curve downward
- At a given detection rate, deeper classifiers have correspondingly higher false positive rates

### 4.1. Training a Cascade of Classifiers

The cascade training process involves two types of trade-offs:

**Trade-offs:**
1. Classifiers with more features achieve higher detection rates and lower false positive rates
2. Classifiers with more features require more time to compute

**Optimization Challenge:** In principle one could define an optimization framework to minimize the expected number of evaluated features by optimizing:
- i) the number of classifier stages
- ii) the number of features in each stage
- iii) the threshold of each stage

Unfortunately, finding this optimum is a tremendously difficult problem.

### Practical Training Framework

A very simple framework is used to produce an effective classifier which is highly efficient:

**Process:**
1. Each stage in the cascade reduces the false positive rate and decreases the detection rate
2. A target is selected for the minimum reduction in false positives and the maximum decrease in detection
3. Each stage is trained by adding features until the target detection and false positive rates are met (determined by testing on a validation set)
4. Stages are added until the overall target for false positive and detection rate is met

### 4.2. Detector Cascade Discussion

**Final Cascade Specifications:**
- 38 stages
- Over 6000 features total
- Fast average detection times due to cascade structure

**Performance on Difficult Dataset:**
- Dataset: 507 faces and 75 million sub-windows
- Average: 10 feature evaluations per sub-window
- About **15 times faster** than Rowley et al. implementation

> **Note:** Henry Rowley very graciously supplied us with implementations of his detection system for direct comparison. Reported results are against his fastest system. The Rowley-Baluja-Kanade detector is widely considered the fastest detection system and has been heavily tested on real-world problems.

### Related Work

**Rowley et al. Face Detection System:**
- Used two detection networks
- A faster yet less accurate network prescreens the image to find candidate regions
- A slower more accurate network processes candidates
- Appears to be the fastest existing face detector (difficult to determine exactly)

**Amit and Geman:**
- Proposed alternative point of view where unusual co-occurrences of simple image features trigger evaluation of more complex detection process
- Full detection process need not be evaluated at many potential image locations and scales
- In their implementation, necessary to first evaluate some feature detector at every location
- These features are then grouped to find unusual co-occurrences
- In practice, our detector and features are so efficient that the amortized cost of evaluating at every scale and location is much faster than finding and grouping edges throughout the image

**Fleuret and Geman:**
- Recent work on face detection using a "chain" of tests
- Image properties measured: disjunctions of fine scale edges (quite different from rectangle features)
- Rectangle features are simple, exist at all scales, and are somewhat interpretable
- Learning philosophy differs radically: density estimation and density discrimination vs. purely discriminative
- False positive rate appears higher than previous approaches
- Paper does not report quantitative results
- Included example images each have between 2 and 10 false positives

---

## 5. Results

A 38 layer cascaded classifier was trained to detect frontal upright faces.

### Training Data

**Face Training Set:**
- 4916 hand labeled faces
- Scaled and aligned to base resolution of 24 by 24 pixels
- Faces extracted from images downloaded during random crawl of the world wide web
- See Figure 5 for typical face examples

**Non-Face Training Set:**
- 9544 images manually inspected and found to not contain any faces
- About 350 million sub-windows within these non-face images

### Cascade Structure

**Number of Features per Layer:**
- Layer 1: 1 feature
- Layer 2: 10 features
- Layer 3: 25 features
- Layer 4: 25 features
- Layer 5: 50 features
- Remaining layers: increasingly more features
- **Total features in all layers: 6061**

### Training Process for Each Layer

**Training Data per Classifier:**
- 4916 training faces (plus vertical mirror images = 9832 total training faces)
- 10,000 non-face sub-windows (size 24 by 24 pixels)
- Using the AdaBoost training procedure

**Non-Face Collection:**
- Initial one feature classifier: non-face training examples collected by selecting random sub-windows from 9544 images without faces
- Subsequent layers: non-face examples obtained by scanning the partial cascade across non-face images and collecting false positives
- Maximum of 10,000 non-face sub-windows collected for each layer

### Speed of the Final Detector

**Performance Metrics:**
- Evaluated on MIT+CMU test set
- Average of **10 features out of 6061** are evaluated per sub-window
- Possible because large majority of sub-windows are rejected by first or second layer
- On 700 MHz Pentium III processor: can process 384 by 288 pixel image in about **0.067 seconds**
- Uses starting scale of 1.25 and step size of 1.5

**Speed Comparisons:**
- Roughly **15 times faster** than Rowley-Baluja-Kanade detector
- About **600 times faster** than Schneiderman-Kanade detector

### Image Processing

**Variance Normalization:**
- All example sub-windows used for training were variance normalized to minimize effect of different lighting conditions
- Normalization is therefore necessary during detection as well

**Computing Variance:**
The variance of an image sub-window can be computed quickly using a pair of integral images.

Formula: σ² = μ² - μ², where:
- σ is the standard deviation
- μ is the mean
- x is the pixel value within the sub-window

**Implementation:**
- Mean of sub-window computed using the integral image
- Sum of squared pixels computed using an integral image of the image squared (two integral images used in scanning process)
- During scanning, effect of image normalization achieved by post-multiplying the feature values rather than pre-multiplying the pixels

### Scanning the Detector

**Scaling:**
- Final detector scanned across the image at multiple scales and locations
- Scaling achieved by scaling the detector itself, rather than scaling the image
- Makes sense because features can be evaluated at any scale with the same cost
- Good results obtained using a set of scales a factor of 1.25 apart

**Location Scanning:**
- Detector also scanned across location
- Subsequent locations obtained by shifting the window some number of pixels Δ
- Shifting process affected by scale of detector: if current scale is s, window is shifted by [s·Δ], where [x] is the rounding operation

**Step Size Effect:**
- Choice of Δ affects both speed and accuracy
- Presented results use Δ = 1.0
- Can achieve significant speedup by setting Δ = 1.5 with only a slight decrease in accuracy

### Integration of Multiple Detections

Since the final detector is insensitive to small changes in translation and scale, multiple detections will usually occur around each face. The same is often true of some types of false positives.

**Postprocessing:**
In practice it makes sense to return one final detection per face by combining overlapping detections.

**Simple Combination Method:**
1. Set of detections first partitioned into disjoint subsets
2. Two detections in same subset if their bounding regions overlap
3. Each partition yields a single final detection
4. Corners of final bounding region are average of corners of all detections in the set

### Experiments on Real-World Test Set

**Test Set:** MIT+CMU frontal face test set
- 130 images
- 507 labeled frontal faces

**ROC Curve Construction:**
- See Figure 6 in paper
- Threshold of final layer classifier adjusted from -∞ to +∞
- Adjusting threshold to -∞ yields: detection rate = 0.0, false positive rate = 0.0
- Adjusting threshold to +∞ increases both rates, but only to a certain point
- Neither rate can be higher than the rate of the detection cascade minus the final layer
- Threshold of +∞ is equivalent to removing that layer
- Further increasing rates requires decreasing threshold of next classifier in cascade
- Complete ROC curve constructed by removing classifier layers

**X-Axis:** Number of false positives (instead of false positive rate) to facilitate comparison with other systems
- To compute false positive rate: divide by total number of sub-windows scanned
- Total sub-windows scanned in experiments: **75,081,800**

### Performance Comparison (Table 2)

Detection rates for various numbers of false positives on MIT+CMU test set (130 images, 507 faces):

| Detector | 10 FP | 31 FP | 50 FP | 65 FP | 78 FP | 95 FP | 167 FP |
|----------|-------|-------|-------|-------|-------|-------|--------|
| Viola-Jones | 76.1% | 88.4% | 91.4% | 92.0% | 92.1% | 92.9% | 93.9% |
| Viola-Jones (voting) | 81.1% | 89.7% | 92.1% | 93.1% | 93.1% | 93.2% | 93.7% |
| Rowley-Baluja-Kanade | 83.2% | 86.0% | - | - | - | 89.2% | 90.1% |
| Schneiderman-Kanade | - | - | - | 94.4% | - | - | - |
| Roth-Yang-Ahuja | - | - | - | - | (94.8%) | - | - |

**Notes:**
- Most previous published results only included single operating regime (single point on ROC curve)
- For Rowley-Baluja-Kanade results: number of different versions tested, all listed under same heading
- For Roth-Yang-Ahuja detector: reported result on MIT+CMU test set minus 5 images containing line drawn faces

**Visual Results:**
See Figure 7 in paper showing output of face detector on test images from MIT+CMU test set.

### Simple Voting Scheme for Improved Results

**Method:**
- Run three detectors (38 layer one described above plus two similarly trained detectors)
- Output the majority vote of the three detectors

**Results:**
- Improves detection rate
- Eliminates more false positives
- Improvement would be greater if detectors were more independent
- Correlation of their errors results in modest improvement over best single detector

---

## 6. Conclusions

We have presented an approach for object detection which minimizes computation time while achieving high detection accuracy. The approach was used to construct a face detection system which is approximately **15 times faster** than any previous approach.

### Contributions

This paper brings together:
- New algorithms
- Representations
- Insights which are quite generic and may well have broader application in computer vision and image processing

### Experimental Validation

This paper presents a set of detailed experiments on a difficult face detection dataset which has been widely studied. This dataset includes faces under a very wide range of conditions including:
- Illumination variation
- Scale variation
- Pose variation
- Camera variation

**Significance:**
- Experiments on such a large and complex dataset are difficult and time consuming
- Systems which work under these conditions are unlikely to be brittle or limited to a single set of conditions
- Conclusions drawn from this dataset are unlikely to be experimental artifacts

---

## References

[1] Y. Amit, D. Geman, and K. Wilder. Joint induction of shape features and tree classifiers, 1997.

[2] Anonymous. Anonymous. In Anonymous, 2000.

[3] F. Crow. Summed-area tables for texture mapping. In Proceedings of SIGGRAPH, volume 18(3), pages 207–212, 1984.

[4] F. Fleuret and D. Geman. Coarse-to-fine face detection. Int. J. Computer Vision, 2001.

[5] William T. Freeman and Edward H. Adelson. The design and use of steerable filters. IEEE Transactions on Pattern Analysis and Machine Intelligence, 13(9):891–906, 1991.

[6] Yoav Freund and Robert E. Schapire. A decision-theoretic generalization of on-line learning and an application to boosting. In Computational Learning Theory: Eurocolt '95, pages 23–37. Springer-Verlag, 1995.

[7] H. Greenspan, S. Belongie, R. Gooodman, P. Perona, S. Rakshit, and C. Anderson. Overcomplete steerable pyramid filters and rotation invariance. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 1994.

[8] L. Itti, C. Koch, and E. Niebur. A model of saliency-based visual attention for rapid scene analysis. IEEE Patt. Anal. Mach. Intell., 20(11):1254–1259, November 1998.

[9] Edgar Osuna, Robert Freund, and Federico Girosi. Training support vector machines: an application to face detection. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 1997.

[10] C. Papageorgiou, M. Oren, and T. Poggio. A general framework for object detection. In International Conference on Computer Vision, 1998.

[11] D. Roth, M. Yang, and N. Ahuja. A snowbased face detector. In Neural Information Processing 12, 2000.

[12] H. Rowley, S. Baluja, and T. Kanade. Neural network-based face detection. In IEEE Patt. Anal. Mach. Intell., volume 20, pages 22–38, 1998.

[13] R. E. Schapire, Y. Freund, P. Bartlett, and W. S. Lee. Boosting the margin: a new explanation for the effectiveness of voting methods. Ann. Stat., 26(5):1651–1686, 1998.

[14] Robert E. Schapire, Yoav Freund, Peter Bartlett, and Wee Sun Lee. Boosting the margin: A new explanation for the effectiveness of voting methods. In Proceedings of the Fourteenth International Conference on Machine Learning, 1997.

[15] H. Schneiderman and T. Kanade. A statistical method for 3D object detection applied to faces and cars. In International Conference on Computer Vision, 2000.

[16] K. Sung and T. Poggio. Example-based learning for view-based face detection. In IEEE Patt. Anal. Mach. Intell., volume 20, pages 39–51, 1998.

[17] J.K. Tsotsos, S.M. Culhane, W.Y.K. Wai, Y.H. Lai, N. Davis, and F. Nuflo. Modeling visual-attention via selective tuning. Artificial Intelligence Journal, 78(1-2):507–545, October 1995.

[18] Andrew Webb. Statistical Pattern Recognition. Oxford University Press, New York, 1999.

---

## Appendix: AdaBoost Algorithm Summary (Table 1)

### Input
- Example images (x₁, y₁), ..., (xₙ, yₙ) where yᵢ = 0,1 for negative and positive examples respectively

### Initialize
- Weights w₁,ᵢ = 1/(2m), 1/(2l) for yᵢ = 0,1 respectively
- Where m and l are the number of negatives and positives respectively

### For t = 1,...,T:

1. **Normalize the weights:**
   - wₜ,ᵢ = wₜ,ᵢ / Σⱼwₜ,ⱼ
   - So that wₜ is a probability distribution

2. **For each feature j, train a classifier hⱼ:**
   - Restricted to using a single feature
   - Error evaluated with respect to wₜ: εⱼ = Σᵢ wᵢ|hⱼ(xᵢ) - yᵢ|

3. **Choose the classifier hₜ with the lowest error εₜ**

4. **Update the weights:**
   - wₜ₊₁,ᵢ = wₜ,ᵢβₜ^(1-eᵢ)
   - Where eᵢ = 0 if example xᵢ is classified correctly, eᵢ = 1 otherwise
   - And βₜ = εₜ/(1-εₜ)

### Final strong classifier:
```
h(x) = {
  1  if Σₜ αₜhₜ(x) ≥ ½Σₜ αₜ
  0  otherwise
}
```
where αₜ = log(1/βₜ)

**Note:** Each round of boosting selects one feature from the 180,000 potential features.