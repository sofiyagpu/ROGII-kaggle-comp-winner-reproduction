# Third-party notices

## Ruby / w5833946 first-place ROGII solution

`notebooks/rogii_winner_reproduction.ipynb` and its generator
`scripts/build_winner_notebook.py` adapt the inference orchestration and conditional ensemble from:

- Author: **Ruby / w5833946**; source repository owner: **IAmAValidUsername**.
- Repository: https://github.com/IAmAValidUsername/kaggle_ROGII_1st_place_solution_Ruby
- Reviewed repository commit: `fade6112ffebd9e27c08ac7f053e8d4c3d5c319a`.
- Notebook: https://www.kaggle.com/code/w5833946/submit-reproduce (API version 2).
- Technical write-up: https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/writeups/1st-place-solution
- Source-code license: **Apache License 2.0**, reproduced below. No separate upstream NOTICE file was present in the reviewed repository.

The original author created the trained model families, model architecture, preprocessing,
geographic reliability gate, and ensemble weights. This repository's adaptations add
isolated checked execution, input-path discovery, scoped temporary storage, no-copy model
loading, final prediction-key validation, runtime test-row coverage checks, sample-order
output, and a provenance report. The unweighted `0802_V2` family is omitted without changing
either final blend. Original inference precision, model configurations (other than runtime
paths and full-test processing), thresholds, and blend order are preserved.

The notebook references these external Kaggle model datasets:

- https://www.kaggle.com/datasets/w5833946/rogii-0719-v1-cv509
- https://www.kaggle.com/datasets/w5833946/rogii-0724-v1-cv486
- https://www.kaggle.com/datasets/w5833946/rogii-0729-v3-cv553
- https://www.kaggle.com/datasets/w5833946/rogii-0801-v1-cv516
- https://www.kaggle.com/datasets/w5833946/rogii-0801-v2-cv480
- https://www.kaggle.com/datasets/w5833946/rogii-0803-v2-cv500

Model weights, author-provided OOF predictions, competition data, and dataset-hosted source
files are not redistributed in this repository. Kaggle reported the dataset license field
as `Unknown` when inspected; the repository's Apache-2.0 license is not a claim that these
separate datasets have been relicensed. Downloaded assets remain subject to their source
terms. The original author explicitly provides the datasets through the public inference notebook.

An archived OOF ensemble score measures the author's released predictions. It is not a new
training result, a reproduced hidden-test score, or a claim that this repository won the competition.
A new submission score may be reported only after Kaggle actually evaluates that run.

## bilzard (tatamikenn) second-place ROGII solution

`src/rogii/second_place.py` and `scripts/predict_second_place.py` run the released AnchorCNN ensemble locally.

- Author: **bilzard** (Kaggle: tatamikenn).
- Weights and inference modules: https://www.kaggle.com/datasets/tatamikenn/rogii-2nd-place-weights
- Inference notebook followed by the adapter: https://www.kaggle.com/code/tatamikenn/rogii-2nd-place-inference
- License: **Apache License 2.0**. The upstream `LICENSE`, `NOTICE`, and `SHA256SUMS` are kept unmodified in `third_party/anchorcnn/`.
- Published private leaderboard RMSE: **5.802**, as stated in the upstream artifact README.

The weights (436 MB), inference modules, and bundled `timm` wheel are not committed. The adapter verifies
every downloaded file against the upstream manifest before importing code or loading weights. The adapter adds
device selection, an MPS decoding path that runs the dynamic program on CPU, sample-order output, and a run report.
Checkpoint selection, phase shifts, and the 1:1 family average follow the upstream notebook.

## Repository structure reference

The user requested the notebook-and-GitHub delivery style of
https://github.com/Spydonny/s6e8-lookup-transformer . Its project layout serves as inspiration;
no model code or weights from that unrelated competition are included here.

## Apache License 2.0 (upstream code)

```text

                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS

   APPENDIX: How to apply the Apache License to your work.

      To apply the Apache License to your work, attach the following
      boilerplate notice, with the fields enclosed by brackets "[]"
      replaced with your own identifying information. (Don't include
      the brackets!)  The text should be enclosed in the appropriate
      comment syntax for the file format. We also recommend that a
      file or class name and description of purpose be included on the
      same "printed page" as the copyright notice for easier
      identification within third-party archives.

   Copyright [yyyy] [name of copyright owner]

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

```
