# Professor Method Section Draft

This note preserves the Overleaf method-section draft used to align the implementation. MoE expert routing and iterative denoising are not part of the current implementation scope.

## Methodology

Figure~\ref{fig:method} summarizes the proposed framework. The method first constructs aspect-specific unsafe drafts from safe mental-health responses. Then, it applies risk-aware masking or deletion to create corrupted drafts at different corruption levels. A shared aspect-conditioned LoRA denoiser is trained to recover the safe response from the question, unsafe draft, aspect vector, and corrupted draft. At inference time, the aspect router estimates the violated counseling dimensions from the question and unsafe draft, and the same denoiser generates the refined safe response.

## Aspect Router and Expert Conditioning

The aspect labels produced during data construction allow the model to learn which counseling dimensions are violated in an unsafe response. We train a router
\[
    H_{\omega}(q,u)=\hat{d} \in [0,1]^K,
\]
where \(\hat{d}_k\) estimates the probability that \(u\) violates aspect \(k\) for question \(q\). The router is trained with a multi-label binary cross-entropy objective:
\[
    \mathcal{L}_{\mathrm{router}}
    =
    \mathbb{E}_{(q,u,d,y)\sim \mathcal{D}}
    \left[
    \sum_{k=1}^{K}
    \left(
    -d_k \log \hat{d}_k
    -
    (1-d_k) \log (1-\hat{d}_k)
    \right)
    \right].
\]

To reduce exposure mismatch between training and inference, the denoiser is conditioned on a mixture of the ground-truth aspect label and the predicted router output. During training, we sample \(m_{\mathrm{tf}} \sim \mathrm{Bernoulli}(p_{\mathrm{tf}})\) and define
\[
    g
    =
    m_{\mathrm{tf}} d
    +
    (1-m_{\mathrm{tf}})\hat{d}.
\]
At inference time, \(g=\hat{d}\) unless an external target aspect vector is supplied.

The same aspect signal is used to activate aspect-specific expert adapters. For each aspect \(k\), we define a low-rank expert update
\[
    \Delta W_k
    =
    \frac{s_{\mathrm{lora}}}{r} B_k A_k,
\]
where \(A_k \in \mathbb{R}^{r \times d_{\mathrm{in}}}\), \(B_k \in \mathbb{R}^{d_{\mathrm{out}} \times r}\), \(r\) is the adapter rank, and \(s_{\mathrm{lora}}\) is a scaling factor. The aspect gate is
\[
    \tau_k(g)
    =
    \frac{g_k+\epsilon}{\sum_{h=1}^{K}(g_h+\epsilon)},
\]
where \(\epsilon>0\) prevents zero division. For a pretrained weight matrix \(W_0\), the adapted matrix is
\[
    W
    =
    W_0
    +
    \Delta W_{\mathrm{sh}}
    +
    \sum_{k=1}^{K} \tau_k(g) \Delta W_k,
\]
where \(\Delta W_{\mathrm{sh}}\) is a shared low-rank update. This gives a mixture of aspect experts while retaining a shared generation backbone.

## Risk-Conditioned Discrete Denoising

We define a task-specific forward corruption process
\[
    q_{\eta}(z_t \mid u,y,g,t),
\]
which constructs a corrupted draft \(z_t\) from the unsafe response \(u\), the safe response \(y\), and the aspect conditioning vector \(g\). The corruption process exposes the model to realistic unsafe-to-safe edit trajectories. Let
\[
    \mathcal{A}(u,y)=\{(a_{\ell},b_{\ell})\}_{\ell=1}^{L}
\]
be a monotonic span alignment between the unsafe response and the safe response. Here, \(a_{\ell}\) is a span from \(u\), and \(b_{\ell}\) is the corresponding span from \(y\). Either span may be empty, which allows the alignment to represent insertion and deletion operations. For each unsafe span \(a_{\ell}\), we compute an aspect-specific span risk score
\[
    r_{\ell}(g)
    =
    \max_{k \in \mathcal{K}}
    g_k R_{\phi,k}(q,a_{\ell}),
\]
where \(R_{\phi,k}(q,a_{\ell}) \in [0,1]\) estimates whether span \(a_{\ell}\) violates aspect \(k\). Higher values of \(r_{\ell}(g)\) indicate spans that are more likely to violate the active counseling aspects.

The corruption strength at timestep \(t\) is controlled by
\[
    \beta_t=\frac{t}{T}.
\]
For each aligned span pair \((a_{\ell},b_{\ell})\), we sample a corruption operation
\[
    c_{\ell}^{(t)} \in \{\mathrm{SAFE}, \mathrm{UNSAFE}, \mathrm{MASK}\}.
\]
The operation probabilities are defined as
\[
    P(c_{\ell}^{(t)}=\mathrm{SAFE}) = 1-\beta_t,
\]
\[
    P(c_{\ell}^{(t)}=\mathrm{MASK}) = \beta_t \pi_{\ell}(t,g),
\]
\[
    P(c_{\ell}^{(t)}=\mathrm{UNSAFE}) = \beta_t (1-\pi_{\ell}(t,g)),
\]
where
\[
    \pi_{\ell}(t,g)
    =
    \operatorname{clip}
    \left(
    \rho_t + \lambda_{\mathrm{mask}} \beta_t r_{\ell}(g),
    0,
    1
    \right).
\]
Here, \(\rho_t\) is the base masking probability at timestep \(t\), and \(\lambda_{\mathrm{mask}}\) controls the strength of risk-aware masking. The span-level realization function is
\[
    \psi(c_{\ell}^{(t)},a_{\ell},b_{\ell})
    =
    \begin{cases}
        b_{\ell}, & c_{\ell}^{(t)}=\mathrm{SAFE}, \\
        a_{\ell}, & c_{\ell}^{(t)}=\mathrm{UNSAFE}, \\
        \langle \mathrm{MASK} \rangle, & c_{\ell}^{(t)}=\mathrm{MASK}.
    \end{cases}
\]
The corrupted draft is obtained by concatenating all realized spans:
\[
    z_t
    =
    \psi(c_{1}^{(t)},a_{1},b_{1})
    \circ
    \cdots
    \circ
    \psi(c_{L}^{(t)},a_{L},b_{L}).
\]
At \(t=0\), the draft is close to the safe response. At larger \(t\), the draft becomes closer to the unsafe response, with high-risk spans preferentially masked. Thus, the forward process defines a discrete edit bridge between unsafe and safe responses.

## Mixture of Corruption Sources

At inference time, the safe response \(y\) is unavailable. To reduce train-test mismatch, we train with a mixture of corruption sources:
\[
    q_{\eta}(z_t \mid u,y,g,t)
    =
    \sum_{b \in \mathcal{B}}
    \omega_b
    q_b(z_t \mid u,y,g,t),
\]
where
\[
    \mathcal{B}
    =
    \{
    \mathrm{bridge},
    \mathrm{unsafe},
    \mathrm{safe},
    \mathrm{empty}
    \},
    \quad
    \sum_{b \in \mathcal{B}} \omega_b = 1.
\]
The bridge corruption \(q_{\mathrm{bridge}}\) is the span-level edit bridge defined above. The unsafe corruption \(q_{\mathrm{unsafe}}\) masks or deletes risky spans in \(u\) and matches the inference setting. The safe corruption \(q_{\mathrm{safe}}\) corrupts spans in \(y\), which provides standard denoising supervision. The empty corruption \(q_{\mathrm{empty}}\) removes the draft variable and reduces to direct supervised refinement. The final training objective therefore includes both draft-conditioned denoising and direct conditional generation.

## Decoder-Backed Reverse Denoiser

The reverse denoising model is parameterized by an autoregressive decoder. Given the serialized condition
\[
    x_t = \mathrm{Serialize}(q,u,g,z_t,t),
\]
the model defines
\[
    p_{\theta}(y \mid q,u,g,z_t,t)
    =
    \prod_{j=1}^{N}
    p_{\theta}
    \left(
    y_j
    \mid
    y_{<j}, q,u,g,z_t,t
    \right).
\]
The decoder predicts the full safe response, so the method supports variable-length rewriting, deletion of unsafe content, insertion of safety guidance, and restructuring of the response. The aspect router selects the expert mixture, and the decoder generates the final safe response conditioned on the selected aspect profile.

## Training Objective

For each training example, we predict an aspect vector, sample an aspect conditioning vector \(g\), sample a timestep \(t\), sample a corrupted draft \(z_t\), and maximize the likelihood of the safe response. The denoising loss is
\[
    \mathcal{L}_{\mathrm{den}}
    =
    \mathbb{E}_{(q,u,d,y)\sim \mathcal{D}}
    \mathbb{E}_{t \sim \mathcal{U}(0,T)}
    \mathbb{E}_{z_t \sim q_{\eta}}
    \left[
    -
    \sum_{j=1}^{N}
    \gamma_j
    \log
    p_{\theta}
    \left(
    y_j
    \mid
    y_{<j},q,u,g,z_t,t
    \right)
    \right].
\]
The token weight \(\gamma_j\) gives additional weight to target tokens associated with high-risk edits:
\[
    \gamma_j
    =
    1
    +
    \lambda_y
    \max_{\ell: y_j \in b_{\ell}}
    r_{\ell}(g),
\]
where \(\lambda_y\) controls the strength of risk-weighted supervision. If no aligned risk score is available for token \(y_j\), we set \(\gamma_j=1\). We also include a direct supervised refinement term:
\[
    \mathcal{L}_{\mathrm{sft}}
    =
    \mathbb{E}_{(q,u,d,y)\sim \mathcal{D}}
    \left[
    -
    \sum_{j=1}^{N}
    \log
    p_{\theta}
    \left(
    y_j
    \mid
    y_{<j},q,u,g
    \right)
    \right].
\]
The final objective is
\[
    \mathcal{L}
    =
    \mathcal{L}_{\mathrm{den}}
    +
    \lambda_{\mathrm{sft}}
    \mathcal{L}_{\mathrm{sft}}
    +
    \lambda_{\mathrm{router}}
    \mathcal{L}_{\mathrm{router}}.
\]
This objective trains the model to reconstruct safe responses from multiple corruption levels, learn aspect-specific violation routing, and preserve the ability to refine directly from the unsafe response.

## Inference

At inference time, the model receives a question \(q\) and an unsafe response \(u\). The router first estimates the violated counseling aspects:
\[
    \hat{d}=H_{\omega}(q,u).
\]
If an external target aspect vector \(d_{\mathrm{ext}}\) is available, the conditioning vector is
\[
    g=\max(\hat{d},d_{\mathrm{ext}}),
\]
where the maximum is applied elementwise. Otherwise, \(g=\hat{d}\). We compute span-level risk scores over the unsafe response and construct a masked unsafe draft
\[
    z_T = C(u,g),
\]
where \(C\) masks or removes spans whose aspect-conditioned risk score exceeds a threshold. The final response is generated by
\[
    \hat{y}
    =
    \arg\max_y
    p_{\theta}
    \left(
    y
    \mid
    q,u,g,z_T,T
    \right).
\]
The primary inference procedure uses a single reverse denoising step from the masked unsafe draft to the final safe response. An iterative variant can also be used by applying the same denoiser across a decreasing sequence of timesteps. Iterative denoising is out of scope for the current v2 data-generation implementation.
