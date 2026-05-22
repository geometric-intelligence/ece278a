import marimo

__generated_with = "0.23.6"
app = marimo.App(layout_file="layouts/SfM.slides.json")


@app.cell(hide_code=True)
def _():
    import importlib
    from pathlib import Path
    import sys

    import cv2
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import plotly.graph_objects as go
    import torch
    from matplotlib.lines import Line2D
    from PIL import Image

    return Image, Line2D, Path, cv2, go, importlib, mo, np, plt, sys, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Structure from Motion (SfM)

    Yanxiu Jin

    Michael Smith

    Jared Arzate

    Anantajit Subrahmanya

    Cameron Cummins
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What is the Structure from Motion (SfM) problem?

    - Recover 3D structure and camera motion from multiple 2D images.
    - Use correspondences of the same points across views as constraints.
    - Output is camera geometry + a 3D point cloud (up to gauge ambiguity).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Affine SfM (Tomasi-Kanade)

    - Assume an affine/orthographic camera so projection is linear.
    - Stack tracked 2D points across frames into a measurement matrix.
    - Factor the matrix into motion and structure, then enforce metric constraints.
    - Affine has its limitations, but we might, e.g., merge multiple partial reconstructions to create a complete pointcloud

    ## COLMAP
    - Industry Standard Practical SfM System
    - Handles non affine case and more
    """)
    return


@app.cell(hide_code=True)
def _(Path, mo):
    frame_dir = Path("data/dino")
    frame_paths = []
    for pattern in (
        "viff.*.ppm",
        "*.ppm",
        "*.png",
        "*.jpg",
        "*.jpeg",
        "*.PNG",
        "*.JPG",
        "*.JPEG",
    ):
        frame_paths.extend(sorted(frame_dir.glob(pattern)))
    frame_paths = sorted(set(frame_paths))
    n_frames = len(frame_paths)
    next_frame = (
        mo.ui.button(
            value=0,
            on_click=lambda i: (i + 1) % n_frames,
            label="Next image",
            kind="neutral",
        )
        if n_frames
        else None
    )
    next_frame
    return frame_paths, n_frames, next_frame


@app.cell(hide_code=True)
def _(Image, frame_paths, mo, n_frames, next_frame):
    idx = next_frame.value
    current = frame_paths[idx]
    mo.vstack(
        [
            mo.md(f"Frame **{idx + 1}/{n_frames}** — `{current.name}`"),
            mo.image(
                Image.open(current).convert("RGB"), width="50%", rounded=True
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Affine SfM

    **First**:

    Feature tracking & Optical Flow
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Optical flow

    Optical flow estimates how image points move between consecutive frames as a 2D velocity field \((u, v)\), assuming brightness constancy and small motion. It reveals temporal correspondence in dynamic scenes.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Optical flow constraint equation
    Let $E(x, y, t)$ be the irradiance at time $t$ at the image point $(x, y)$.
    $u(x, y)$ and $v(x, y)$ are the $x$ and $y$ components of the optical flow vector at that point, we expect that the irradiance will be the same at time
    $t + \delta t$ at the point $(x + \delta x, y + \delta y)$, where $\delta x = u \delta t$
    and $\delta y = v \delta t$. That is,

    $$E(x + u\,\delta t,\, y + v\,\delta t,\, t + \delta t) = E(x, y, t)$$

    for a small time interval $\delta t$. This single constraint is not sufficient to
    determine both $u$ and $v$ uniquely.

    If brightness varies smoothly with $x$, $y$, and $t$, we can expand the left-hand
    side of the equation above in a Taylor series and so obtain

    $$E(x, y, t) + \delta x \frac{\partial E}{\partial x} + \delta y \frac{\partial E}{\partial y} + \delta t \frac{\partial E}{\partial t} + e = E(x, y, t),$$

    where $e$ contains second- and higher-order terms in $\delta x$, $\delta y$, and $\delta t$.

    $$\frac{\partial E}{\partial x}\frac{dx}{dt} + \frac{\partial E}{\partial y}\frac{dy}{dt} + \frac{\partial E}{\partial t} = 0,$$

    which is actually just the expansion of the equation

    $$\frac{dE}{dt} = 0$$

    in the total derivative of $E$ with respect to time. Using the abbreviations

    $$u = \frac{dx}{dt}, \qquad v = \frac{dy}{dt},$$

    $$E_x = \frac{\partial E}{\partial x}, \quad E_y = \frac{\partial E}{\partial y}, \quad E_t = \frac{\partial E}{\partial t},$$

    we obtain

    $$E_x u + E_y v + E_t = 0.$$

    The spatial and temporal gradient $E_x$, $E_y$, and $E_t$ are estimated from the image. The above
    equation is called the *optical flow constraint equation*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Aperture problem
    Consider a two-dimensional space with axes $u$ and $v$, which we shall call
    *velocity space*. Values of $(u, v)$ satisfying the constraint equation
    lie on a straight line in velocity space. All that a local measurement can do is
    to identify this constraint line.

    $$(E_x, E_y) \cdot (u, v) = -E_t.$$

    The component of optical flow in the direction of the brightness gradient
    $(E_x, E_y)^T$ is thus

    $$\frac{E_t}{\sqrt{E_x^2 + E_y^2}}.$$

    We cannot determine the component of the optical flow at right
    angles to this direction. This
    ambiguity is also known as the *aperture problem*.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Lucas-Kanade method
    Lucas-Kanade added additional constraints to recover full flow vectors by assuming nearly constant flow in a small patch.
    i.e. Assume that u,v are constant over a small neighborhood. For N pixels in a local window,
    $$I_x(x_i)u + I_y(x_i)v = -I_t(x_i), \quad i = 1, \ldots, N$$

    This gives the linear system:

    $$A \begin{bmatrix} u \\ v \end{bmatrix} = \mathbf{b}$$

    with:

    $$A = \begin{bmatrix} I_x(x_1) & I_y(x_1) \\ \vdots & \vdots \\ I_x(x_N) & I_y(x_N) \end{bmatrix}, \quad \mathbf{b} = -\begin{bmatrix} I_t(x_1) \\ \vdots \\ I_t(x_N) \end{bmatrix}$$

    The least squares solution is:

    $$\begin{bmatrix} u \\ v \end{bmatrix} = (A^T A)^{-1} A^T \mathbf{b}$$
    """)
    return


@app.cell(hide_code=True)
def _(Image, cv2, frame_paths, mo, np):
    first_idx, second_idx = 7, 8
    first = np.array(Image.open(frame_paths[first_idx]).convert("RGB"))
    second = np.array(Image.open(frame_paths[second_idx]).convert("RGB"))

    gray_first = cv2.cvtColor(first, cv2.COLOR_RGB2GRAY)
    gray_second = cv2.cvtColor(second, cv2.COLOR_RGB2GRAY)
    flow_init = np.zeros(
        (gray_first.shape[0], gray_first.shape[1], 2), dtype=np.float32
    )
    flow = cv2.calcOpticalFlowFarneback(
        gray_first, gray_second, flow_init, 0.5, 3, 15, 3, 5, 1.2, 0
    )

    flow_vis = first.copy()
    step = 18
    for y in range(step // 2, gray_first.shape[0], step):
        for x in range(step // 2, gray_first.shape[1], step):
            dx, dy = flow[y, x]
            p0 = (x, y)
            p1 = (int(x + dx), int(y + dy))
            cv2.arrowedLine(flow_vis, p0, p1, (0, 255, 0), 1, tipLength=0.3)

    mo.hstack(
        [
            mo.vstack(
                [
                    mo.md(f"**Frame 1** (`{frame_paths[first_idx].name}`)"),
                    mo.image(Image.fromarray(first), rounded=True),
                ]
            ),
            mo.vstack(
                [
                    mo.md(f"**Frame 2** (`{frame_paths[second_idx].name}`)"),
                    mo.image(Image.fromarray(second), rounded=True),
                ]
            ),
            mo.vstack(
                [
                    mo.md("**Optical flow**"),
                    mo.image(Image.fromarray(flow_vis), rounded=True),
                    #                mo.md("_Legend: green arrows show motion direction and size._"),
                ]
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Feature tracking

    - Detect salient points and follow the same points across frames.
        - SIFT for extracting keypoints
        - Optical Flow (Lucas-Kanade method) for tracking
    - Keep only trajectories visible in many frames for stability.
    - Reject obvious outliers/drift before building the matrix.
    """)
    return


@app.cell(hide_code=True)
def _(Image, Line2D, cv2, frame_paths, np, plt):
    selected_indices = [i for i in (7, 8, 9) if i < len(frame_paths)]
    if len(selected_indices) < 2:
        selected_indices = list(range(min(3, len(frame_paths))))

    selected = [frame_paths[i] for i in selected_indices]
    rgbs = [np.array(Image.open(p).convert("RGB")) for p in selected]
    grays = [cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) for rgb in rgbs]

    sift = cv2.SIFT_create(nfeatures=400, contrastThreshold=0.08)  # type: ignore[attr-defined]
    kps = [sift.detectAndCompute(g, None)[0] for g in grays]
    pts = [np.array([kp.pt for kp in ks], dtype=np.float32) for ks in kps]

    if len(grays) == 0:
        tracked = []
    elif len(pts[0]) == 0:
        tracked = [np.empty((0, 2), dtype=np.float32) for _ in grays]
    else:
        tracked_points = [pts[0].reshape(-1, 1, 2)]
        for _i in range(1, len(grays)):
            p_prev = tracked_points[-1]
            if len(p_prev) == 0:
                tracked_points.append(p_prev.copy())
                continue

            p_next, s, _ = cv2.calcOpticalFlowPyrLK(
                grays[_i - 1], grays[_i], p_prev, p_prev.copy()
            )
            if p_next is None or s is None:
                tracked_points = [
                    np.empty((0, 1, 2), dtype=np.float32) for _ in tracked_points
                ]
                tracked_points.append(np.empty((0, 1, 2), dtype=np.float32))
                continue

            p_back, sb, _ = cv2.calcOpticalFlowPyrLK(
                grays[_i], grays[_i - 1], p_next, p_next.copy()
            )
            if p_back is None or sb is None:
                tracked_points = [
                    np.empty((0, 1, 2), dtype=np.float32) for _ in tracked_points
                ]
                tracked_points.append(np.empty((0, 1, 2), dtype=np.float32))
                continue

            p_next = p_next.astype(np.float32)
            p_back = p_back.astype(np.float32)
            fb = np.linalg.norm((p_back - p_prev).reshape(-1, 2), axis=1)
            ok = (
                s.reshape(-1).astype(bool)
                & sb.reshape(-1).astype(bool)
                & (fb < 1.0)
            )

            tracked_points = [p[ok] for p in tracked_points]
            tracked_points.append(p_next[ok])

        tracked = [p.reshape(-1, 2) for p in tracked_points]

    if len(selected) == 0:
        _fig, _ax = plt.subplots(figsize=(6, 4))
        _ax.text(
            0.5,
            0.5,
            "No frames available for tracking",
            ha="center",
            va="center",
        )
        _ax.axis("off")
        _fig

    _fig, _ax = plt.subplots(1, len(selected), figsize=(5 * len(selected), 5))
    if len(selected) == 1:
        _ax = [_ax]
    legend = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="red",
            markersize=6,
            label="SIFT keypoints",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="blue",
            markersize=7,
            label="Tracked points",
        ),
    ]
    for _i, p in enumerate(selected):
        _ax[_i].imshow(rgbs[_i])
        if len(pts[_i]):
            _ax[_i].scatter(pts[_i][:, 0], pts[_i][:, 1], s=5, c="red")
        if len(tracked[_i]):
            _ax[_i].scatter(tracked[_i][:, 0], tracked[_i][:, 1], s=7, c="blue")
        _ax[_i].set_title(f"{p.name} ({len(kps[_i])}, {len(tracked[0])} tracked)")
        _ax[_i].axis("off")
    _ax[-1].legend(handles=legend, loc="lower right", framealpha=0.9)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Affine approximation
    We've tracked P feature points across F images $(x_{fp}, y_{fp})$ for frame f and point p. We want spatial information $X_p$,$Y_p$,$Z_p$

    Under an affine camera model (weak perspective) then we get something like $x = AX + t$


    $$ x_{fp} = m_{f}^T \begin{bmatrix}
    X_{p} \\
    Y_{p} \\
    Z_{p}
    \end{bmatrix} + t_f $$

    Where $m_f = \in \mathbb{R^{1 \times 3}}$ is from camera.

    This approximation is good if depth variation is small compared to distance from camera (approximately linear in Z). Will see later that for more general SfM, affine approximation is not good.


    <!--
    - Build \(W \in \mathbb{R}^{2F 	imes P}\) from tracked \((x, y)\) coordinates.
    - Each frame contributes two rows (x-row and y-row).
    - Columns represent a single 3D point trajectory across frames.
    -->
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The Measurement Matrix

    For one frame we have
    $$
    \begin{bmatrix}
    x_{f1} & x_{f2} & \cdots & x_{fP}  \\
    y_{f1} & y_{f2} & \cdots & y_{fP}
    \end{bmatrix}
    = \begin{bmatrix}
    a_{f}^T  \\
    b_{f}^T
    \end{bmatrix}
    \begin{bmatrix}
    X_{1} & X_{2} & \cdots & X_P \\
    \end{bmatrix}
    +
    \begin{bmatrix}
    t_{xf}  \\
    t_{yf} \\
    \end{bmatrix} \mathbb{I}^T
    $$

    We **center** by subtracting the frame's centroid from all tracked points. This removes per-frame translation from the affine model and is crucial later.

    This allows us to write a matrix for this frame $W_f$ as

    $$
    W_f =
    \begin{bmatrix}
    a_{f}^T  \\
    b_{f}^T \\
    \end{bmatrix} S
    $$

    Where S is all the P 3D points.

    We stack all frames vertically and let $M = \begin{bmatrix}
    a_{1}^T  \\
    b_{1}^T \\
    a_{2}^T  \\
    \vdots
    \end{bmatrix} S$

    Then the measurement matrix $W = MS$, $W \in \mathbb{R}^{2F\times P}, M \in \mathbb{R}^{2F \times 3}, S \in \mathbb{R}^{3 \times P}$.
    Each image coordinate is linear function of 3D point.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Tomasi & Kanade Factorization

    Tomasi and Kanade (1992) observed that under affine projection, the centered mesaurement matrix has at *most* rank 3. Our derivation holds then up to affine ambigutity.

    From SVD, we have $W = U \Sigma V^T$. Tomasi and Kanade showed that the factorization will always connect motion and 3D shape, that is, for rank 3 approximation:

    $$M = U_3 \Sigma_{3}^{1/2},  S = \Sigma_{3}^{1/2} V_{3}^T $$

    $$W = MS \in \mathbb{R}^{2F \times P}$$

    In the end, each column of $W$ is a 3P point tracked over time, and each pair of rows is a camera frame.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Last step: Metric upgrade

    - Factorization is not unique (any invertible affine transform works)
    - Solve for a transform that enforces orthonormality constraints on motion rows (each row of camera matrix is orthogonal and equal magnitude)
    """)
    return


@app.cell(hide_code=True)
def _(Image, Path, cv2, np, plt):
    import plotly.graph_objects as plt_g
    from scipy.linalg import sqrtm


    def main():
        frame_paths = sorted(Path("./data/dino").glob("*.ppm"))
        selected_paths = frame_paths[:10]

        rgb_frames, frames = load_images(selected_paths)

        tracked, F = get_keypoints(frames)

        W = construct_W(tracked, F)

        plot_tracks(tracked, rgb_frames, F)

        M, S = affineSFM(W, F)

        # remove outliers
        dist = np.linalg.norm(S, axis=0)
        mask = dist < np.percentile(dist, 90)
        S = S[:, mask]

        plot_shape(S)


    def construct_W(tracks, F):
        P = len(tracks)
        W = np.zeros((2 * F, P))
        for j, track in enumerate(tracks):
            for i, pt in enumerate(track):
                x, y = pt
                W[2 * i, j] = x
                W[2 * i + 1, j] = y

        # centering
        for i in range(F):
            W[2 * i] -= np.mean(W[2 * i])
            W[2 * i + 1] -= np.mean(W[2 * i + 1])

        return W


    def affineSFM(W, F):
        Ud, Sd, Vd = np.linalg.svd(W)

        # keep up to rank 3
        U3 = Ud[:, 0:3]
        S3 = np.diag(Sd[:3])
        V3 = Vd.T[:, :3]

        M = U3 @ sqrtm(S3)
        S = sqrtm(S3) @ V3.T

        C = metric_upgrade(F, M)

        M = M @ C
        S = np.linalg.inv(C) @ S

        return M, S


    def plot_shape(S):
        x = S[0, :]
        y = S[1, :]
        z = S[2, :]

        fig = plt_g.Figure(
            data=[
                plt_g.Scatter3d(
                    x=x,
                    y=y,
                    z=z,
                    mode="markers",
                    marker=dict(size=4, color=z, colorscale="Turbo", opacity=0.9),
                )
            ]
        )

        fig.update_layout(
            title="Affine SfM Reconstruction",
            scene=dict(aspectmode="data", dragmode="orbit"),
        )

        fig.show()

        fig = plt.figure(figsize=(18.5, 6))
        fig.suptitle("Reconstructed image")
        ax1 = fig.add_subplot(131, projection="3d")
        ax1.scatter(x, y, z, color="red", lw=1)
        ax1.view_init(130, 0)
        ax2 = fig.add_subplot(132, projection="3d")
        ax2.scatter(x, y, z, color="red", lw=1)
        ax2.view_init(45, 180)
        ax3 = fig.add_subplot(133, projection="3d")
        ax3.scatter(x, y, z, color="red", lw=1)
        ax3.view_init(-90, 90)
        plt.show()


    def get_keypoints(frames):
        MAX_CORNERS = 800
        QUALITY = 0.01
        MIN_DISTANCE = 8

        LK_PARAMS = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )

        ## define a mask to eliminate background junk
        mask = np.zeros_like(frames[0])
        mask[10:490, 10:500] = 255

        p0 = cv2.goodFeaturesToTrack(
            frames[0],
            mask=mask,
            maxCorners=MAX_CORNERS,
            qualityLevel=QUALITY,
            minDistance=MIN_DISTANCE,
        )

        tracks = [[pt.ravel()] for pt in p0]

        prev_pts = p0
        for i in range(1, len(frames)):
            prev_img = frames[i - 1]
            curr_img = frames[i]

            next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_img, curr_img, prev_pts, None, **LK_PARAMS
            )

            good_new = next_pts[status == 1]
            _ = prev_pts[status == 1]

            new_tracks = []

            idx = 0
            for t, s in zip(tracks, status):
                if s == 1:
                    t.append(good_new[idx])
                    new_tracks.append(t)
                    idx += 1

            tracks = new_tracks

            prev_pts = good_new.reshape(-1, 1, 2)

        tracks = [t for t in tracks if len(t) == len(frames)]

        F = len(frames)

        return tracks, F


    def metric_upgrade(F, M):

        def _symmetric_vec(a, b):
            return np.array(
                [
                    a[0] * b[0],
                    a[0] * b[1] + a[1] * b[0],
                    a[0] * b[2] + a[2] * b[0],
                    a[1] * b[1],
                    a[1] * b[2] + a[2] * b[1],
                    a[2] * b[2],
                ]
            )

        def _positiveDef(M):
            """
            method to compute nearest positive definite matrix
            """
            M = (M + M.T) * 0.5
            k = 0
            I = np.eye(M.shape[0])
            while True:
                try:
                    _ = np.linalg.cholesky(M)
                    break
                except np.linalg.LinAlgError:
                    k += 1
                    _, v = np.linalg.eig(M)
                    min_eig = v.min()
                    M += (-min_eig * k * k + np.spacing(min_eig)) * I
            return M

        A = []
        b = []

        for i in range(F):
            ai = M[2 * i]
            bi = M[2 * i + 1]

            # ai^T L ai = 1
            A.append(_symmetric_vec(ai, ai))
            b.append(1)

            # bi^T L bi = 1
            A.append(_symmetric_vec(bi, bi))
            b.append(1)

            # ai^T L bi = 0
            A.append(_symmetric_vec(ai, bi))
            b.append(0)

        A = np.array(A)
        b = np.array(b)

        x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

        L = np.array([[x[0], x[1], x[2]], [x[1], x[3], x[4]], [x[2], x[4], x[5]]])

        L = _positiveDef(L)

        C = np.linalg.cholesky(L)
        return C


    def load_images(paths):
        rgb_frames = [np.array(Image.open(p).convert("RGB")) for p in paths]
        frames = []
        for path in paths:
            img = cv2.imread(path)
            if img is None:
                raise RuntimeError(f"Failed to load {path}")
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            frames.append(gray)
        return rgb_frames, frames


    def plot_tracks(tracked, rgbs, F):
        P = len(tracked)

        tracked = np.array(tracked)

        tracked = np.transpose(tracked, (1, 0, 2))

        _, ax = plt.subplots(1, min(3, F), figsize=(15, 5))

        if F == 1:
            ax = [ax]

        show_idx = [0, F // 2, F - 1]

        for k, idx in enumerate(show_idx):
            ax[k].imshow(rgbs[idx])

            x = tracked[idx, :, 0]
            y = tracked[idx, :, 1]

            ax[k].scatter(x, y, s=8, c="cyan")

            ax[k].set_title(f"Frame {idx} ({P} tracked)")

            ax[k].axis("off")

        plt.tight_layout()
        plt.show()


    if __name__ == "__main__":
        main()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Beyond Affine: COLMAP

    **Industry Standard Practical SfM System**

    COLMAP is the practical version of the SfM pipeline we have been discussing.

    Compared with affine Tomasi--Kanade, COLMAP:

    - works with **perspective projection**, not just affine projection
    - accepts **ordered or unordered** image collections
    - does not assume all images overlap
    - detects and matches features automatically
    - performs geometric verification, triangulation, and bundle adjustment

    In this section, we use **PyCOLMAP** for the actual COLMAP pipeline and small visual demos to explain each concept.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## COLMAP: Practical Structure from Motion

    COLMAP turns a set of unordered images into a **sparse 3D reconstruction** by solving for camera geometry and scene structure simultaneously.

    The full pipeline is:

    $$
    \underbrace{\mathcal{I}_1, \mathcal{I}_2, \ldots, \mathcal{I}_n}_{\text{images}}
    \xrightarrow{\text{extract}}
    \underbrace{\{(k_i, d_i)\}}_{\text{features}}
    \xrightarrow{\text{match}}
    \underbrace{\mathcal{C}_{ij}}_{\text{candidates}}
    \xrightarrow{\text{verify}}
    \underbrace{\mathcal{M}_{ij}}_{\text{inliers}}
    \xrightarrow{\text{triangulate}}
    \underbrace{\mathbf{X}_j \in \mathbb{R}^3}_{\text{3D points}}
    \xrightarrow{\text{refine}}
    \underbrace{\{\hat{P}_i, \hat{\mathbf{X}}_j\}}_{\text{reconstruction}}
    $$

    Each arrow is a distinct algorithmic step. We cover all five below, with both visual demos (OpenCV) and the real pipeline (PyCOLMAP).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Calling COLMAP from Python

    PyCOLMAP exposes the full COLMAP pipeline as three function calls:

    ```python
    pycolmap.extract_features(database_path, image_dir)   # step 1
    pycolmap.match_exhaustive(database_path)               # step 2
    maps = pycolmap.incremental_mapping(                   # steps 3–5
        database_path, image_dir, output_dir
    )
    ```

    Internally, `incremental_mapping` registers images one at a time, triangulates new 3D points after each registration, and runs **bundle adjustment** to keep reprojection error low throughout.

    The surrounding cells use OpenCV to visualise each sub-step in isolation.
    """)
    return


@app.cell
def _():
    try:
        import pycolmap as pycolmap_module

        colmap_pycolmap_available = True
        colmap_pycolmap_error = None
        colmap_pycolmap_version = getattr(
            pycolmap_module, "__version__", "unknown"
        )
    except Exception as _pycolmap_exc:
        pycolmap_module = None
        colmap_pycolmap_available = False
        colmap_pycolmap_error = str(_pycolmap_exc)
        colmap_pycolmap_version = None

    print("PyCOLMAP available:", colmap_pycolmap_available)
    print("PyCOLMAP version:", colmap_pycolmap_version)
    if not colmap_pycolmap_available:
        print("PyCOLMAP import error:", colmap_pycolmap_error)
        print("Install with: pip install pycolmap")
    return colmap_pycolmap_available, pycolmap_module


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## COLMAP demo dataset: Sacré-Cœur

    The initial SfM/Tomasi--Kanade section still uses the Dino sequence.

    For the COLMAP section, we switch to a more realistic unordered image collection:

    ```text
    data/sacre_coeur/*.jpg
    ```

    This is closer to how COLMAP is normally used: a folder of overlapping perspective images of the same scene.

    We use two Sacré-Cœur images for the visual feature/matching demos, and the full folder subset for the actual PyCOLMAP reconstruction.
    """)
    return


@app.cell(hide_code=True)
def _():
    from pathlib import Path as _ColmapPath

    _colmap_sacre_dir = _ColmapPath("data/building_front")

    colmap_image_paths = sorted(_colmap_sacre_dir.glob("*.jpg"))
    colmap_image_paths += sorted(_colmap_sacre_dir.glob("*.jpeg"))
    colmap_image_paths += sorted(_colmap_sacre_dir.glob("*.png"))
    colmap_image_paths += sorted(_colmap_sacre_dir.glob("*.JPG"))
    colmap_image_paths += sorted(_colmap_sacre_dir.glob("viff.*.ppm"))

    # Pick two images for the visual demos. Use nearby entries if possible,
    # otherwise fall back to the first two images.
    if len(colmap_image_paths) >= 3:
        colmap_img1_path = colmap_image_paths[0]
        colmap_img2_path = colmap_image_paths[1]
    elif len(colmap_image_paths) >= 2:
        colmap_img1_path = colmap_image_paths[0]
        colmap_img2_path = colmap_image_paths[1]
    else:
        colmap_img1_path = None
        colmap_img2_path = None

    print("Sacré-Cœur image folder:", _colmap_sacre_dir)
    print("Number of Sacré-Cœur images:", len(colmap_image_paths))
    print("COLMAP demo image 1:", colmap_img1_path)
    print("COLMAP demo image 2:", colmap_img2_path)

    if len(colmap_image_paths) < 2:
        print("Expected images in: data/sacre_coeur/*.jpg")
    return colmap_image_paths, colmap_img1_path, colmap_img2_path


@app.cell(hide_code=True)
def _(Image, colmap_image_paths, plt):
    _colmap_n_preview = min(8, len(colmap_image_paths))

    if _colmap_n_preview > 0:
        _colmap_cols = min(4, _colmap_n_preview)
        _colmap_rows = (_colmap_n_preview + _colmap_cols - 1) // _colmap_cols
        colmap_fig_dataset, _colmap_axes_dataset = plt.subplots(
            _colmap_rows,
            _colmap_cols,
            figsize=(4 * _colmap_cols, 3 * _colmap_rows),
        )

        if _colmap_n_preview == 1:
            _colmap_axes_flat = [_colmap_axes_dataset]
        else:
            _colmap_axes_flat = list(_colmap_axes_dataset.ravel())

        for _colmap_ax_i, _colmap_path_i in zip(
            _colmap_axes_flat, colmap_image_paths[:_colmap_n_preview]
        ):
            _colmap_img_i = Image.open(_colmap_path_i).convert("RGB")
            _colmap_ax_i.imshow(_colmap_img_i)
            _colmap_ax_i.set_title(_colmap_path_i.name, fontsize=9)
            _colmap_ax_i.axis("off")

        for _colmap_ax_i in _colmap_axes_flat[_colmap_n_preview:]:
            _colmap_ax_i.axis("off")

        colmap_fig_dataset.suptitle(
            "Sacré-Cœur images used for the COLMAP section"
        )
        colmap_fig_dataset.tight_layout()
    else:
        colmap_fig_dataset, _colmap_ax_dataset = plt.subplots(figsize=(7, 4))
        _colmap_ax_dataset.text(
            0.5,
            0.5,
            "No images found in data/sacre_coeur",
            ha="center",
            va="center",
        )
        _colmap_ax_dataset.axis("off")

    colmap_fig_dataset
    return


@app.cell(hide_code=True)
def _(Image, colmap_img1_path, colmap_img2_path, cv2, np):
    if colmap_img1_path is not None and colmap_img2_path is not None:
        colmap_img1_rgb = np.array(Image.open(colmap_img1_path).convert("RGB"))
        colmap_img2_rgb = np.array(Image.open(colmap_img2_path).convert("RGB"))

        colmap_img1_gray = cv2.cvtColor(colmap_img1_rgb, cv2.COLOR_RGB2GRAY)
        colmap_img2_gray = cv2.cvtColor(colmap_img2_rgb, cv2.COLOR_RGB2GRAY)
    else:
        colmap_img1_rgb = None
        colmap_img2_rgb = None
        colmap_img1_gray = None
        colmap_img2_gray = None
    return colmap_img1_gray, colmap_img1_rgb, colmap_img2_gray, colmap_img2_rgb


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 1 — Feature Extraction

    Each image $\mathcal{I}_i$ is mapped to a set of *keypoints* with associated descriptors:

    $$
    \mathcal{I}_i \;\longmapsto\; \bigl\{\,(k_{ij},\, \mathbf{d}_{ij})\bigr\}_{j=1}^{N_i}
    $$

    where $k_{ij} = (u, v, \sigma, \theta)$ encodes **location**, **scale** $\sigma$, and **orientation** $\theta$, and $\mathbf{d}_{ij} \in \mathbb{R}^{128}$ is the SIFT descriptor.

    **Why scale and orientation matter.** SIFT builds its descriptor in a *normalised patch* centred on $k_{ij}$: the patch is rotated by $-\theta$ and resized to a fixed scale. This makes $\mathbf{d}_{ij}$ invariant to in-plane rotation and scale change — the same physical point yields nearly identical descriptors across very different viewpoints.

    Circles in the plot below encode $\sigma$ (radius) and $\theta$ (tick direction).
    """)
    return


@app.cell(hide_code=True)
def _(colmap_img1_gray, colmap_img1_rgb, cv2, plt):
    if colmap_img1_gray is not None and colmap_img1_rgb is not None:
        try:
            _colmap_detector_feature = cv2.SIFT_create(nfeatures=500)
            colmap_detector_name = "SIFT"
        except Exception:
            _colmap_detector_feature = cv2.ORB_create(nfeatures=500)
            colmap_detector_name = "ORB"

        colmap_kp1, colmap_desc1 = _colmap_detector_feature.detectAndCompute(
            colmap_img1_gray,
            None,
        )

        _colmap_keypoint_vis = cv2.drawKeypoints(
            colmap_img1_rgb,
            colmap_kp1,
            None,
            color=(0, 255, 80),
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        )
        # Overdraw a filled dot at each center so small keypoints are visible
        for _kp in colmap_kp1:
            _cx, _cy = int(_kp.pt[0]), int(_kp.pt[1])
            cv2.circle(_colmap_keypoint_vis, (_cx, _cy), 8, (255, 80, 0), -1)

        colmap_fig_feature, _colmap_ax_feature = plt.subplots(figsize=(14, 10))
        _colmap_ax_feature.imshow(_colmap_keypoint_vis)
        _colmap_ax_feature.set_title(
            f"Step 1: Feature Extraction ({colmap_detector_name})"
        )
        _colmap_ax_feature.axis("off")
    else:
        colmap_detector_name = "None"
        colmap_kp1 = []
        colmap_desc1 = None

        colmap_fig_feature, _colmap_ax_feature = plt.subplots(figsize=(14, 10))
        _colmap_ax_feature.text(
            0.5,
            0.5,
            "No image loaded",
            ha="center",
            va="center",
        )
        _colmap_ax_feature.axis("off")

    print("Detector:", colmap_detector_name)
    print("Number of keypoints in image 1:", len(colmap_kp1))
    print(
        "Descriptor shape:", None if colmap_desc1 is None else colmap_desc1.shape
    )

    colmap_fig_feature
    return colmap_desc1, colmap_detector_name, colmap_kp1


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 2 — Feature Matching

    Given descriptors from two images, candidate correspondences are found by nearest-neighbour search in descriptor space:

    $$
    k_{1p} \;\leftrightarrow\; k_{2q}
    \quad \text{if} \quad
    \frac{\|\mathbf{d}_{1p} - \mathbf{d}_{2q}\|}{\|\mathbf{d}_{1p} - \mathbf{d}_{2q'}\|} < \tau
    $$

    This is **Lowe's ratio test**: a match is accepted only when the nearest neighbour is significantly closer than the second-nearest neighbour $q'$. A threshold of $\tau = 0.75$ is standard — it rejects ambiguous matches where two descriptors are similarly plausible.

    These are still *candidate* correspondences. Many will be outliers caused by repeated textures, reflections, or descriptor collisions. Geometric verification (next step) filters them.
    """)
    return


@app.cell(hide_code=True)
def _(
    colmap_desc1,
    colmap_detector_name,
    colmap_img1_rgb,
    colmap_img2_gray,
    colmap_img2_rgb,
    colmap_kp1,
    cv2,
    plt,
):
    if (
        colmap_img1_rgb is not None
        and colmap_img2_rgb is not None
        and colmap_img2_gray is not None
        and colmap_desc1 is not None
        and len(colmap_kp1) > 0
    ):
        if colmap_detector_name == "SIFT":
            _colmap_detector_match = cv2.SIFT_create(nfeatures=500)
            colmap_kp2, colmap_desc2 = _colmap_detector_match.detectAndCompute(
                colmap_img2_gray,
                None,
            )

            _colmap_matcher = cv2.BFMatcher(cv2.NORM_L2)
            _colmap_raw_knn_matches = _colmap_matcher.knnMatch(
                colmap_desc1,
                colmap_desc2,
                k=2,
            )

            colmap_matches = []
            for _colmap_pair in _colmap_raw_knn_matches:
                if len(_colmap_pair) == 2:
                    _colmap_m, _colmap_n = _colmap_pair
                    if _colmap_m.distance < 0.75 * _colmap_n.distance:
                        colmap_matches.append(_colmap_m)
        else:
            _colmap_detector_match = cv2.ORB_create(nfeatures=500)
            colmap_kp2, colmap_desc2 = _colmap_detector_match.detectAndCompute(
                colmap_img2_gray,
                None,
            )

            _colmap_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            colmap_matches = _colmap_matcher.match(colmap_desc1, colmap_desc2)
            colmap_matches = sorted(colmap_matches, key=lambda _m: _m.distance)

        colmap_matches = colmap_matches[:100]

        # Assign a unique color per match ranked by descriptor distance
        _n_show = min(40, len(colmap_matches))
        _sorted_matches = sorted(colmap_matches, key=lambda m: m.distance)[
            :_n_show
        ]

        _colmap_match_vis = cv2.drawMatches(
            colmap_img1_rgb,
            colmap_kp1,
            colmap_img2_rgb,
            colmap_kp2,
            _sorted_matches,
            None,
            matchColor=(0, 255, 120),  # bright green lines
            singlePointColor=(80, 80, 80),
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )

        # Overdraw thicker lines and endpoint dots manually for visibility
        _h, _w = colmap_img1_rgb.shape[:2]
        for _m in _sorted_matches:
            _p1 = tuple(map(int, colmap_kp1[_m.queryIdx].pt))
            _p2 = (
                int(colmap_kp2[_m.trainIdx].pt[0]) + _w,
                int(colmap_kp2[_m.trainIdx].pt[1]),
            )
            cv2.line(
                _colmap_match_vis, _p1, _p2, (0, 230, 255), 4
            )  # cyan, thickness=2
            cv2.circle(
                _colmap_match_vis, _p1, 5, (255, 80, 0), -1
            )  # orange dot left
            cv2.circle(
                _colmap_match_vis, _p2, 5, (255, 80, 0), -1
            )  # orange dot right

        colmap_fig_match, _colmap_ax_match = plt.subplots(figsize=(24, 10))
        _colmap_ax_match.imshow(_colmap_match_vis)
        _colmap_ax_match.set_title(
            f"Step 2: Candidate Feature Matches ({len(colmap_matches)} shown/kept)"
        )
        _colmap_ax_match.axis("off")
    else:
        colmap_kp2 = []
        colmap_desc2 = None
        colmap_matches = []

        colmap_fig_match, _colmap_ax_match = plt.subplots(figsize=(24, 10))
        _colmap_ax_match.text(
            0.5,
            0.5,
            "Feature matching unavailable\nRun feature extraction first.",
            ha="center",
            va="center",
        )
        _colmap_ax_match.axis("off")

    print("Keypoints in image 1:", len(colmap_kp1))
    print("Keypoints in image 2:", len(colmap_kp2))
    print("Candidate matches:", len(colmap_matches))

    colmap_fig_match
    return colmap_kp2, colmap_matches


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 3 — Geometric Verification

    Two cameras viewing the same 3D point must satisfy the **epipolar constraint**. For corresponding image points $\mathbf{x} \in \mathcal{I}_1$ and $\mathbf{x}' \in \mathcal{I}_2$:

    $$
    \mathbf{x}'^{\top} \mathbf{F} \mathbf{x} = 0
    $$

    $\mathbf{F} \in \mathbb{R}^{3 \times 3}$ is the **fundamental matrix** — rank 2, seven degrees of freedom. It encodes the complete relative geometry between the two cameras without knowing their intrinsics.

    When intrinsics $\mathbf{K}$ are known, $\mathbf{F}$ factors into the **essential matrix** $\mathbf{E} = \mathbf{K}'^{\top}\mathbf{F}\mathbf{K}$, which has only five degrees of freedom and decomposes into a relative rotation and translation:

    $$
    \mathbf{E} = [\mathbf{t}]_\times \mathbf{R}, \qquad \mathbf{E} = \mathbf{U}\,\text{diag}(1,1,0)\,\mathbf{V}^\top
    $$

    **RANSAC** estimates $\mathbf{F}$ (or $\mathbf{E}$) robustly: it repeatedly samples the minimum number of point pairs (7 for $\mathbf{F}$, 5 for $\mathbf{E}$), fits the matrix, and counts inliers whose epipolar distance falls below a pixel threshold $\epsilon$.
    """)
    return


@app.cell(hide_code=True)
def _(colmap_kp1, colmap_kp2, colmap_matches, cv2, np):
    if len(colmap_matches) >= 8:
        colmap_pts1 = np.float32(
            [colmap_kp1[_m.queryIdx].pt for _m in colmap_matches]
        )
        colmap_pts2 = np.float32(
            [colmap_kp2[_m.trainIdx].pt for _m in colmap_matches]
        )

        colmap_F, colmap_inlier_mask = cv2.findFundamentalMat(
            colmap_pts1,
            colmap_pts2,
            method=cv2.FM_RANSAC,
            ransacReprojThreshold=1.0,
            confidence=0.99,
        )

        if colmap_inlier_mask is not None:
            colmap_inlier_mask = colmap_inlier_mask.ravel().astype(bool)
            colmap_inlier_matches = [
                _m
                for _m, _keep in zip(colmap_matches, colmap_inlier_mask)
                if _keep
            ]
        else:
            colmap_inlier_matches = []
    else:
        colmap_pts1 = np.empty((0, 2))
        colmap_pts2 = np.empty((0, 2))
        colmap_F = None
        colmap_inlier_mask = np.array([], dtype=bool)
        colmap_inlier_matches = []

    print("Candidate matches:", len(colmap_matches))
    print("Geometrically verified inliers:", len(colmap_inlier_matches))
    print("Estimated fundamental matrix:")
    print(colmap_F)
    return colmap_F, colmap_inlier_matches


@app.cell(hide_code=True)
def _(
    colmap_img1_rgb,
    colmap_img2_rgb,
    colmap_inlier_matches,
    colmap_kp1,
    colmap_kp2,
    cv2,
    plt,
):
    if colmap_img1_rgb is not None and colmap_img2_rgb is not None:
        import numpy as _np_match

        _n_show = min(40, len(colmap_inlier_matches))
        _sorted_inlier_matches = sorted(
            colmap_inlier_matches, key=lambda m: m.distance
        )[:_n_show]

        _colmap_verified_vis = cv2.drawMatches(
            colmap_img1_rgb,
            colmap_kp1,
            colmap_img2_rgb,
            colmap_kp2,
            _sorted_inlier_matches,
            None,
            matchColor=(0, 230, 255),  # cyan lines
            singlePointColor=(80, 80, 80),
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )

        # Overdraw thicker lines manually
        _h, _w = colmap_img1_rgb.shape[:2]
        for _m in _sorted_inlier_matches:
            _p1 = tuple(map(int, colmap_kp1[_m.queryIdx].pt))
            _p2 = (
                int(colmap_kp2[_m.trainIdx].pt[0]) + _w,
                int(colmap_kp2[_m.trainIdx].pt[1]),
            )
            cv2.line(_colmap_verified_vis, _p1, _p2, (0, 230, 255), 2)

        colmap_fig_verify, _colmap_ax_verify = plt.subplots(figsize=(24, 10))
        _colmap_ax_verify.imshow(_colmap_verified_vis)
        _colmap_ax_verify.set_title(
            f"Step 3: Geometrically Verified Matches ({len(colmap_inlier_matches)} inliers)",
            fontsize=18,
        )
        _colmap_ax_verify.axis("off")
    else:
        colmap_fig_verify, _colmap_ax_verify = plt.subplots(figsize=(24, 10))
        _colmap_ax_verify.text(
            0.5, 0.5, "No images loaded", ha="center", va="center"
        )
        _colmap_ax_verify.axis("off")

    colmap_fig_verify
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 4 — Triangulation

    Given camera projection matrices $\mathbf{P}_1, \mathbf{P}_2$ and verified 2D correspondences $(\mathbf{x}_1, \mathbf{x}_2)$, we recover a 3D point $\mathbf{X}$ from the pair of ray equations:

    $$
    \mathbf{x}_1 \times (\mathbf{P}_1 \mathbf{X}) = \mathbf{0}, \qquad
    \mathbf{x}_2 \times (\mathbf{P}_2 \mathbf{X}) = \mathbf{0}
    $$

    Stacking rows from each cross-product gives an overdetermined linear system $\mathbf{A}\mathbf{X} = \mathbf{0}$ (the **DLT**). The solution minimising $\|\mathbf{A}\mathbf{X}\|$ subject to $\|\mathbf{X}\|=1$ is the last right singular vector of $\mathbf{A}$:

    $$
    \hat{\mathbf{X}} = \arg\min_{\mathbf{X}} \|\mathbf{A}\mathbf{X}\|_2
    $$

    Two rays in $\mathbb{R}^3$ generically *skew* — they do not intersect exactly due to noise. The DLT solution minimises the algebraic residual; **optimal triangulation** minimises the reprojection error directly, which is a non-linear problem solved via the Sampson approximation.
    """)
    return


@app.cell(hide_code=True)
def _(
    colmap_F,
    colmap_img1_gray,
    colmap_inlier_matches,
    colmap_kp1,
    colmap_kp2,
    cv2,
    np,
):
    if (
        colmap_F is not None
        and len(colmap_inlier_matches) >= 8
        and colmap_img1_gray is not None
    ):
        _colmap_h, _colmap_w = colmap_img1_gray.shape

        colmap_focal = 0.9 * max(_colmap_h, _colmap_w)
        colmap_K = np.array(
            [
                [colmap_focal, 0, _colmap_w / 2],
                [0, colmap_focal, _colmap_h / 2],
                [0, 0, 1],
            ],
            dtype=float,
        )

        colmap_pts1_in = np.float32(
            [colmap_kp1[_m.queryIdx].pt for _m in colmap_inlier_matches]
        )
        colmap_pts2_in = np.float32(
            [colmap_kp2[_m.trainIdx].pt for _m in colmap_inlier_matches]
        )

        colmap_E, colmap_E_mask = cv2.findEssentialMat(
            colmap_pts1_in,
            colmap_pts2_in,
            colmap_K,
            method=cv2.RANSAC,
            threshold=1.0,
            prob=0.999,
        )

        if colmap_E is not None:
            _colmap_pose_ok, colmap_R, colmap_t, colmap_pose_mask = (
                cv2.recoverPose(
                    colmap_E,
                    colmap_pts1_in,
                    colmap_pts2_in,
                    colmap_K,
                )
            )

            _colmap_P1 = colmap_K @ np.hstack([np.eye(3), np.zeros((3, 1))])
            _colmap_P2 = colmap_K @ np.hstack([colmap_R, colmap_t])

            _colmap_points4D = cv2.triangulatePoints(
                _colmap_P1,
                _colmap_P2,
                colmap_pts1_in.T,
                colmap_pts2_in.T,
            )

            colmap_points3D = (_colmap_points4D[:3] / _colmap_points4D[3]).T
        else:
            colmap_R = np.eye(3)
            colmap_t = np.zeros((3, 1))
            colmap_pose_mask = None
            colmap_points3D = np.empty((0, 3))
    else:
        colmap_K = None
        colmap_E = None
        colmap_E_mask = None
        colmap_R = np.eye(3)
        colmap_t = np.zeros((3, 1))
        colmap_pose_mask = None
        colmap_pts1_in = np.empty((0, 2))
        colmap_pts2_in = np.empty((0, 2))
        colmap_points3D = np.empty((0, 3))

    print("Triangulated 3D points:", colmap_points3D.shape)
    return (
        colmap_K,
        colmap_R,
        colmap_points3D,
        colmap_pts1_in,
        colmap_pts2_in,
        colmap_t,
    )


@app.cell(hide_code=True)
def _(colmap_points3D, np, plt):
    colmap_fig_tri = plt.figure(figsize=(24, 10))
    _colmap_ax_tri = colmap_fig_tri.add_subplot(111, projection="3d")

    if len(colmap_points3D) > 0:
        _colmap_pts3d_vis = colmap_points3D.copy()
        _colmap_finite = np.isfinite(_colmap_pts3d_vis).all(axis=1)
        _colmap_pts3d_vis = _colmap_pts3d_vis[_colmap_finite]

        if len(_colmap_pts3d_vis) > 0:
            _colmap_norms = np.linalg.norm(_colmap_pts3d_vis, axis=1)
            _colmap_keep = _colmap_norms < np.percentile(_colmap_norms, 90)
            _colmap_pts3d_vis = _colmap_pts3d_vis[_colmap_keep]

            _colmap_ax_tri.scatter(
                _colmap_pts3d_vis[:, 0],
                _colmap_pts3d_vis[:, 1],
                _colmap_pts3d_vis[:, 2],
                s=10,
            )
            _colmap_ax_tri.set_title("Step 4: Triangulated Sparse 3D Points")
            _colmap_ax_tri.set_xlabel("X")
            _colmap_ax_tri.set_ylabel("Y")
            _colmap_ax_tri.set_zlabel("Z")
        else:
            _colmap_ax_tri.text(0.5, 0.5, 0.5, "No stable 3D points")
    else:
        _colmap_ax_tri.text(0.5, 0.5, 0.5, "Triangulation unavailable")

    colmap_fig_tri.tight_layout()
    colmap_fig_tri
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 5 — Bundle Adjustment

    All previous steps introduce drift. Bundle adjustment (BA) is the gold-standard joint refinement: it simultaneously optimises **all** camera poses, **all** intrinsic parameters, and **all** 3D point positions by minimising total reprojection error:

    $$
    \min_{\{\mathbf{R}_i, \mathbf{t}_i, \mathbf{K}_i\},\, \{\mathbf{X}_j\}}
    \sum_{(i,j) \in \mathcal{V}}
    \rho\!\left(\left\|\mathbf{x}_{ij} - \pi(\mathbf{K}_i,\, \mathbf{R}_i,\, \mathbf{t}_i,\, \mathbf{X}_j)\right\|^2\right)
    $$

    where $\pi$ is the perspective projection function, $\mathcal{V}$ is the set of visible point-image pairs, and $\rho$ is a **robust kernel** (e.g. Cauchy or Huber) that downweights residuals from outlier observations.


    COLMAP runs BA after every image registration, keeping the growing reconstruction numerically healthy.
    """)
    return


@app.cell
def _():
    # The problem is solved with the **Levenberg–Marquardt** algorithm. The key computational insight is that the Jacobian has a *sparse block structure*: each 3D point only appears in the cameras that observe it. Exploiting this with the **Schur complement** reduces the $O((6m + 3n)^3)$ naive solve to roughly $O(m^3)$, where $m$ is the number of cameras and $n \gg m$ is the number of 3D points.
    return


@app.cell(hide_code=True)
def _(
    colmap_K,
    colmap_R,
    colmap_points3D,
    colmap_pts1_in,
    colmap_pts2_in,
    colmap_t,
    cv2,
    np,
    plt,
):
    if colmap_K is not None and len(colmap_points3D) > 0:
        _colmap_n = min(
            len(colmap_points3D),
            len(colmap_pts1_in),
            len(colmap_pts2_in),
        )

        _colmap_X = colmap_points3D[:_colmap_n]
        _colmap_obs1 = colmap_pts1_in[:_colmap_n]
        _colmap_obs2 = colmap_pts2_in[:_colmap_n]

        _colmap_rvec1 = np.zeros((3, 1))
        _colmap_tvec1 = np.zeros((3, 1))
        _colmap_proj1, _ = cv2.projectPoints(
            _colmap_X,
            _colmap_rvec1,
            _colmap_tvec1,
            colmap_K,
            None,
        )
        _colmap_proj1 = _colmap_proj1.reshape(-1, 2)

        _colmap_rvec2, _ = cv2.Rodrigues(colmap_R)
        _colmap_proj2, _ = cv2.projectPoints(
            _colmap_X,
            _colmap_rvec2,
            colmap_t,
            colmap_K,
            None,
        )
        _colmap_proj2 = _colmap_proj2.reshape(-1, 2)

        _colmap_err1 = np.linalg.norm(_colmap_proj1 - _colmap_obs1, axis=1)
        _colmap_err2 = np.linalg.norm(_colmap_proj2 - _colmap_obs2, axis=1)

        colmap_reprojection_errors = np.concatenate([_colmap_err1, _colmap_err2])
        colmap_mean_reprojection_error = float(np.mean(colmap_reprojection_errors))
    else:
        colmap_reprojection_errors = np.array([])
        colmap_mean_reprojection_error = None

    colmap_fig_ba, _colmap_ax_ba = plt.subplots(figsize=(12, 6))

    if len(colmap_reprojection_errors) > 0:
        _colmap_ax_ba.hist(colmap_reprojection_errors, bins=30)
        _colmap_ax_ba.axvline(
            colmap_mean_reprojection_error,
            linestyle="--",
            label="Mean error",
        )
        _colmap_ax_ba.set_title("Step 5: Reprojection Error")
        _colmap_ax_ba.set_xlabel("Reprojection error in pixels")
        _colmap_ax_ba.set_ylabel("Count")
        _colmap_ax_ba.legend()
    else:
        _colmap_ax_ba.text(
            0.5,
            0.5,
            "No reprojection errors available",
            ha="center",
            va="center",
        )
        _colmap_ax_ba.axis("off")

    print("Mean reprojection error:", colmap_mean_reprojection_error)

    colmap_fig_ba.tight_layout()
    colmap_fig_ba
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Actual PyCOLMAP Reconstruction Demo

    The previous cells visualize the individual ideas.

    This cell runs the actual PyCOLMAP pipeline if PyCOLMAP is installed:

    1. `extract_features`
    2. `match_exhaustive`
    3. `incremental_mapping`

    This is the closest part of the notebook to real COLMAP.

    Set the image folder and max image count, then press the button to run it.

    By default, the folder is:

    ```text
    data/sacre_coeur
    ```

    It may take a little time depending on the image count and machine.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    colmap_dataset_path_input = mo.ui.text(
        value="data/sacre_coeur",
        label="Image folder",
        placeholder="Path to images",
    )
    colmap_max_images_input = mo.ui.number(
        value=30,
        start=2,
        step=1,
        label="Max images",
    )
    # colmap_run_pycolmap_button = mo.ui.button(
    #     value=0,
    #     on_click=lambda value: value + 1,
    #     label="Run actual PyCOLMAP sparse reconstruction",
    #     kind="success",
    # )

    mo.vstack(
        [
            mo.hstack([colmap_dataset_path_input, colmap_max_images_input]),
            # colmap_run_pycolmap_button,
        ]
    )
    return colmap_dataset_path_input, colmap_max_images_input


@app.cell(hide_code=True)
def _(colmap_dataset_path_input, colmap_max_images_input):
    from pathlib import Path as _ColmapPath
    import shutil as _colmap_shutil

    colmap_source_dir = _ColmapPath(colmap_dataset_path_input.value).expanduser()
    colmap_source_image_paths = []

    if colmap_source_dir.exists():
        for _pattern in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.ppm"):
            colmap_source_image_paths += sorted(colmap_source_dir.glob(_pattern))
    else:
        print("Input image folder does not exist:", colmap_source_dir)

    colmap_max_images = int(colmap_max_images_input.value)

    colmap_pycolmap_workspace = _ColmapPath("pycolmap_sacre_coeur_workspace")
    colmap_pycolmap_image_dir = colmap_pycolmap_workspace / "images"
    colmap_pycolmap_database_path = colmap_pycolmap_workspace / "database.db"
    colmap_pycolmap_sparse_dir = colmap_pycolmap_workspace / "sparse"

    # Recreate the image folder each run so stale files from another dataset do not remain.
    colmap_pycolmap_workspace.mkdir(exist_ok=True)
    if colmap_pycolmap_image_dir.exists():
        _colmap_shutil.rmtree(colmap_pycolmap_image_dir)
    colmap_pycolmap_image_dir.mkdir(exist_ok=True)
    colmap_pycolmap_sparse_dir.mkdir(exist_ok=True)

    # Use a manageable subset for a live demo.
    colmap_pycolmap_images_used = colmap_source_image_paths[:colmap_max_images]

    for _colmap_src in colmap_pycolmap_images_used:
        _colmap_dst = colmap_pycolmap_image_dir / _colmap_src.name
        _colmap_shutil.copy2(_colmap_src, _colmap_dst)

    print("PyCOLMAP workspace:", colmap_pycolmap_workspace)
    print("Source image directory:", colmap_source_dir)
    print("Source images found:", len(colmap_source_image_paths))
    print("PyCOLMAP image directory:", colmap_pycolmap_image_dir)
    print(
        "Images copied for reconstruction:",
        len(list(colmap_pycolmap_image_dir.glob("*"))),
    )
    return (
        colmap_pycolmap_database_path,
        colmap_pycolmap_image_dir,
        colmap_pycolmap_images_used,
        colmap_pycolmap_sparse_dir,
    )


@app.cell(hide_code=True)
def _(
    colmap_pycolmap_available,
    colmap_pycolmap_database_path,
    colmap_pycolmap_image_dir,
    colmap_pycolmap_images_used,
    colmap_pycolmap_sparse_dir,
    np,
    pycolmap_module,
):
    colmap_pycolmap_status = "not_run"
    colmap_pycolmap_reconstruction = None
    colmap_pycolmap_points_xyz = np.empty((0, 3))
    colmap_pycolmap_num_images = 0
    colmap_pycolmap_num_points3D = 0

    if len(colmap_pycolmap_images_used) < 2:
        colmap_pycolmap_status = "not_enough_images"
        print("Need at least 2 images in the selected folder.")
    elif not colmap_pycolmap_available:
        colmap_pycolmap_status = "pycolmap_not_installed"
        print("PyCOLMAP is not installed. Install with: pip install pycolmap")
    else:
        try:
            import shutil as _colmap_shutil

            if colmap_pycolmap_database_path.exists():
                colmap_pycolmap_database_path.unlink()

            if colmap_pycolmap_sparse_dir.exists():
                _colmap_shutil.rmtree(colmap_pycolmap_sparse_dir)
            colmap_pycolmap_sparse_dir.mkdir(exist_ok=True)

            print("Running pycolmap.extract_features(...) with more SIFT features")

            _colmap_feature_options = pycolmap_module.FeatureExtractionOptions()
            _colmap_feature_options.max_image_size = 1200
            _colmap_feature_options.num_threads = 8
            _colmap_feature_options.use_gpu = True

            _colmap_feature_options.sift.max_num_features = 4096
            _colmap_feature_options.sift.peak_threshold = 0.006
            _colmap_feature_options.sift.edge_threshold = 10.0
            _colmap_feature_options.use_gpu = True

            pycolmap_module.extract_features(
                colmap_pycolmap_database_path,
                colmap_pycolmap_image_dir,
                extraction_options=_colmap_feature_options,
            )

            print("Running pycolmap.match_exhaustive(...)")
            pycolmap_module.match_exhaustive(colmap_pycolmap_database_path)

            print("Running pycolmap.incremental_mapping(...)")
            _colmap_maps = pycolmap_module.incremental_mapping(
                colmap_pycolmap_database_path,
                colmap_pycolmap_image_dir,
                colmap_pycolmap_sparse_dir,
            )

            if _colmap_maps:
                _colmap_keys = list(_colmap_maps.keys())
                _colmap_first_key = sorted(_colmap_keys)[0]
                colmap_pycolmap_reconstruction = _colmap_maps[_colmap_first_key]

                colmap_pycolmap_num_images = len(
                    colmap_pycolmap_reconstruction.images
                )
                colmap_pycolmap_num_points3D = len(
                    colmap_pycolmap_reconstruction.points3D
                )

                _colmap_xyz_list = []
                for (
                    _colmap_point
                ) in colmap_pycolmap_reconstruction.points3D.values():
                    _colmap_xyz_list.append(_colmap_point.xyz)

                if _colmap_xyz_list:
                    colmap_pycolmap_points_xyz = np.array(_colmap_xyz_list)

                colmap_pycolmap_status = "success"
            else:
                colmap_pycolmap_status = "no_reconstruction"

        except Exception as _colmap_pycolmap_exc:
            colmap_pycolmap_status = f"error: {type(_colmap_pycolmap_exc).__name__}: {_colmap_pycolmap_exc}"

    print("PyCOLMAP status:", colmap_pycolmap_status)
    print("Registered images:", colmap_pycolmap_num_images)
    print("Sparse 3D points:", colmap_pycolmap_num_points3D)
    return colmap_pycolmap_reconstruction, colmap_pycolmap_status


@app.cell(hide_code=True)
def _(colmap_pycolmap_reconstruction, colmap_pycolmap_status, mo, np):
    # interactive rotation/zoom, RGB-colored sparse points, and camera centers.
    try:
        import plotly.graph_objects as _colmap_go

        _colmap_plotly_available = True
        _colmap_plotly_error = None
    except Exception as _colmap_plotly_exc:
        _colmap_plotly_available = False
        _colmap_plotly_error = str(_colmap_plotly_exc)

    if not _colmap_plotly_available:
        colmap_fig_pycolmap = mo.md(
            f"""
            !!! warning "Plotly is not installed"

                Install it with:

                ```bash
                uv add plotly
                ```

                Import error: `{_colmap_plotly_error}`
            """
        )
    elif (
        colmap_pycolmap_status != "success"
        or colmap_pycolmap_reconstruction is None
    ):
        colmap_fig_pycolmap = mo.md(
            f"""
            !!! warning "PyCOLMAP reconstruction unavailable"

                Status: `{colmap_pycolmap_status}`
            """
        )
    else:
        _colmap_xyz_list = []
        _colmap_color_list = []

        for _colmap_point in colmap_pycolmap_reconstruction.points3D.values():
            _colmap_xyz = np.asarray(_colmap_point.xyz, dtype=float)
            if not np.isfinite(_colmap_xyz).all():
                continue

            _colmap_color = getattr(_colmap_point, "color", None)
            if _colmap_color is None:
                _colmap_color = getattr(_colmap_point, "rgb", None)

            if _colmap_color is not None:
                _colmap_rgb = np.asarray(_colmap_color).astype(int).clip(0, 255)
                _colmap_color_str = (
                    f"rgb({_colmap_rgb[0]},{_colmap_rgb[1]},{_colmap_rgb[2]})"
                )
            else:
                _colmap_color_str = "rgb(40,120,220)"

            _colmap_xyz_list.append(_colmap_xyz)
            _colmap_color_list.append(_colmap_color_str)

        if _colmap_xyz_list:
            _colmap_points_xyz = np.vstack(_colmap_xyz_list)
            _colmap_colors = np.array(_colmap_color_list, dtype=object)

            # Remove extreme outliers for a cleaner presentation view.
            _colmap_center = np.median(_colmap_points_xyz, axis=0)
            _colmap_dist = np.linalg.norm(
                _colmap_points_xyz - _colmap_center, axis=1
            )
            _colmap_keep = _colmap_dist < np.percentile(_colmap_dist, 98)
            _colmap_points_xyz = _colmap_points_xyz[_colmap_keep]
            _colmap_colors = _colmap_colors[_colmap_keep]
        else:
            _colmap_points_xyz = np.empty((0, 3))
            _colmap_colors = []

        _colmap_camera_centers = []
        _colmap_camera_names = []

        for _colmap_image in colmap_pycolmap_reconstruction.images.values():
            try:
                _colmap_has_pose = getattr(_colmap_image, "has_pose", True)
                if callable(_colmap_has_pose):
                    _colmap_has_pose = _colmap_has_pose()
                if not _colmap_has_pose:
                    continue

                _colmap_cam_from_world = getattr(
                    _colmap_image, "cam_from_world", None
                )
                if callable(_colmap_cam_from_world):
                    _colmap_cam_from_world = _colmap_cam_from_world()

                _colmap_world_from_cam = _colmap_cam_from_world.inverse()
                _colmap_center_xyz = np.asarray(
                    _colmap_world_from_cam.translation, dtype=float
                )
            except Exception:
                # Older pycolmap versions expose a direct projection_center method.
                try:
                    _colmap_center_xyz = np.asarray(
                        _colmap_image.projection_center(), dtype=float
                    )
                except Exception:
                    continue

            if np.isfinite(_colmap_center_xyz).all():
                _colmap_camera_centers.append(_colmap_center_xyz)
                _colmap_camera_names.append(
                    getattr(_colmap_image, "name", "camera")
                )

        if _colmap_camera_centers:
            _colmap_camera_centers = np.vstack(_colmap_camera_centers)
        else:
            _colmap_camera_centers = np.empty((0, 3))

        colmap_fig_pycolmap = _colmap_go.Figure()

        if len(_colmap_points_xyz) > 0:
            colmap_fig_pycolmap.add_trace(
                _colmap_go.Scatter3d(
                    x=_colmap_points_xyz[:, 0],
                    y=_colmap_points_xyz[:, 1],
                    z=_colmap_points_xyz[:, 2],
                    mode="markers",
                    name="Sparse 3D points",
                    marker=dict(
                        size=4,
                        color=_colmap_colors,
                        opacity=0.85,
                    ),
                    hoverinfo="skip",
                )
            )

        if len(_colmap_camera_centers) > 0:
            colmap_fig_pycolmap.add_trace(
                _colmap_go.Scatter3d(
                    x=_colmap_camera_centers[:, 0],
                    y=_colmap_camera_centers[:, 1],
                    z=_colmap_camera_centers[:, 2],
                    mode="markers+lines",
                    name="Registered cameras",
                    marker=dict(size=5, color="red", symbol="diamond"),
                    line=dict(color="red", width=3),
                    text=_colmap_camera_names,
                    hovertemplate="%{text}<extra></extra>",
                )
            )

        colmap_fig_pycolmap.update_layout(
            title=(
                "Actual PyCOLMAP Sparse Reconstruction "
                f"({_colmap_points_xyz.shape[0]} points, "
                f"{_colmap_camera_centers.shape[0]} cameras)"
            ),
            width=1300,
            height=1000,
            scene=dict(
                aspectmode="data",
                dragmode="orbit",
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z",
                xaxis=dict(
                    showbackground=True, backgroundcolor="rgb(245,245,245)"
                ),
                yaxis=dict(
                    showbackground=True, backgroundcolor="rgb(245,245,245)"
                ),
                zaxis=dict(
                    showbackground=True, backgroundcolor="rgb(245,245,245)"
                ),
            ),
            legend=dict(x=0.02, y=0.98),
            margin=dict(l=0, r=0, t=50, b=0),
        )

    colmap_fig_pycolmap
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## COLMAP Summary

    | Step | What it does | Demo in this notebook |
    |---|---|---|
    | Feature extraction | Detect repeatable local features | Keypoint visualization |
    | Feature matching | Match descriptors across images | Candidate match visualization |
    | Geometric verification | Reject matches that violate camera geometry | RANSAC inliers using the fundamental matrix |
    | Triangulation | Estimate 3D points from verified matches | Two-view sparse point demo |
    | Bundle adjustment | Refine cameras and 3D points | Reprojection error objective |
    | PyCOLMAP | Runs real COLMAP pipeline from Python | `extract_features`, `match_exhaustive`, `incremental_mapping` |

    The OpenCV cells explain the concepts visually.

    The PyCOLMAP cell is the actual COLMAP-based reconstruction pipeline.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # VGGT: Visual Geometry Grounded Transformer
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Why **not** COLMAP (always, everywhere, all the time)?

    - Optimization of Visual Geometry is a computationally intensive task

      - Recent papers (DUSt3R, MASt3R, VGGSfM) has demonstrated learning-based approaches but do not automate the full pipeline

    - COLMAP takes 15s while VGGT takes <1s
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## VGGT Architecture

    **VGGT is a single feed-forward transformer**

    Given some images in the scene, it predicts the camera matrices, depth maps, 3D point maps and dense point cloud.

    ![VGGT Architecture Diagram](https://vgg-t.github.io/resources/architecture_v4.png)

    ### Pipeline Steps

    1. Patchification, tokenization for each image  (using DINOv2)

    2. Append 1 camera token + 4 register tokens

    3. Transformer blocks - alternating between frame-wise self attention and global self-attention

    4. Output tokens - camera head contains intrinsics + extrinsics, DPT head (image tokens) contains depth maps, point maps, tracking features
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Vision Transformers (in a nutshell)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Training and Losses

    VGGT is trained end-to-end with dense geometric supervision from posed multi-view data which is explicitly labeled.

    **Training data:** large-scale Internet and synthetic multi-view datasets with camera poses and depth/geometry supervision (sampled as image sets from the same scene).

    A compact view of the objective is:
    \[
    \mathcal{L}_{\text{total}} =
    \lambda_{\text{cam}}\,\mathcal{L}_{\text{cam}} +
    \lambda_{\text{depth}}\,\mathcal{L}_{\text{depth}} +
    \lambda_{\text{pts}}\,\mathcal{L}_{\text{3D}} +
    \lambda_{\text{track}}\,\mathcal{L}_{\text{track}}.
    \]

    where:
    \[
    \begin{aligned}
    \mathcal{L}_{\text{cam}}\; &=\; \alpha_R\,\left\lVert \log\!\left(R\hat{R}^{\top}\right) \right\rVert_1 + \alpha_t\,\left\lVert t-\hat{t} \right\rVert_1 + \alpha_K\,\left\lVert K-\hat{K} \right\rVert_1 \\
    \mathcal{L}_{\text{depth}} &=\; \frac{1}{|\Omega|}\sum_{p\in\Omega} \rho\!\left(\log d(p)-\log \hat d(p)\right) \\
    \mathcal{L}_{\text{3D}}\; &=\; \frac{1}{|\Omega|}\sum_{p\in\Omega} \rho\!\left(\|X(p)-\hat X(p)\|_2\right) \\
    \mathcal{L}_{\text{track}} &=\; \frac{1}{|\mathcal{M}|}\sum_{(p,q)\in\mathcal{M}} \rho\!\left(\|\pi_j(\hat X_i(p)) - q\|_2\right)
    \end{aligned}
    \]
    with \(\rho\) representing some robust penalty function (reducing the impact of outliers), \(\Omega\) valid pixels, and \(\mathcal{M}\) cross-view matches.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## VGGT demo: Dino reconstruction (CUDA)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    data_dir = mo.ui.text(value="data/sacre_coeur", label="Image folder")
    data_limit = mo.ui.number(value=30, start=2, step=1, label="Max images")
    point_density = mo.ui.slider(
        start=1000,
        stop=120000,
        step=1000,
        value=10000,
        label="Point density",
    )
    mo.vstack(
        [
            mo.hstack([mo.md("**VGGT data path**"), data_dir], justify="start"),
            mo.hstack(
                [mo.md("**Input image limit**"), data_limit], justify="start"
            ),
            mo.hstack(
                [mo.md("**Displayed points**"), point_density], justify="start"
            ),
        ]
    )
    return data_dir, data_limit, point_density


@app.cell(hide_code=True)
def _(Image, Path, data_dir, data_limit, importlib, np, sys, torch):
    _device = torch.device("cuda")

    # import hacks to bring in VGGT into a notebook
    _local_vggt_root = str((Path.cwd() / "vggt").resolve())
    if _local_vggt_root not in sys.path:
        sys.path.insert(0, _local_vggt_root)
    _VGGT = importlib.import_module("vggt.models.vggt").VGGT
    _pose_encoding_to_extri_intri = importlib.import_module(
        "vggt.utils.pose_enc"
    ).pose_encoding_to_extri_intri
    _unproject_depth_map_to_point_map = importlib.import_module(
        "vggt.utils.geometry"
    ).unproject_depth_map_to_point_map

    # extract image paths
    _image_root = Path(data_dir.value)
    _image_paths = []
    for _pattern in (
        "*.ppm",
        "*.png",
        "*.jpg",
        "*.jpeg",
        "*.PPM",
        "*.PNG",
        "*.JPG",
        "*.JPEG",
    ):
        _image_paths.extend(sorted(_image_root.glob(_pattern)))
    _image_paths = sorted(set(_image_paths))
    assert _image_paths, f"No images found in {data_dir.value} (ppm/png/jpg/jpeg)"

    _limit = max(2, int(data_limit.value))
    _image_paths = _image_paths[:_limit]

    _take_idx = np.linspace(
        0, len(_image_paths) - 1, min(12, len(_image_paths)), dtype=int
    )
    _sample_paths = [_image_paths[i] for i in _take_idx]

    # extract image frames
    _raw_frames = [
        np.array(Image.open(p).convert("RGB"), dtype=np.uint8)
        for p in _sample_paths
    ]
    _h = min(img.shape[0] for img in _raw_frames)
    _w = min(img.shape[1] for img in _raw_frames)
    # crop down to a patch size of 14x14
    _h = (_h // 14) * 14
    _w = (_w // 14) * 14
    _rgb_frames = [
        img[
            (img.shape[0] - _h) // 2 : (img.shape[0] + _h) // 2,
            (img.shape[1] - _w) // 2 : (img.shape[1] + _w) // 2,
        ]
        for img in _raw_frames
    ]
    _images = (
        torch.from_numpy(np.stack(_rgb_frames))
        .permute(0, 3, 1, 2)
        .float()
        .to(_device)
        / 255.0
    )

    # load model onto GPU
    _model = _VGGT.from_pretrained("facebook/VGGT-1B").eval().to(_device)
    with torch.inference_mode():
        # get prediction vectors -- technically, after this, we are done
        _pred = _model(_images)

    # post processing -- extract points from the model predictions
    _extr, _intr = _pose_encoding_to_extri_intri(
        _pred["pose_enc"], _images.shape[-2:]
    )
    _points3d = _unproject_depth_map_to_point_map(
        _pred["depth"][0], _extr[0], _intr[0]
    )

    points3d = _points3d.reshape(-1, 3)
    colors = _images.permute(0, 2, 3, 1).detach().cpu().numpy().reshape(-1, 3)
    _valid = np.isfinite(points3d).all(axis=1)
    points3d, colors = points3d[_valid], colors[_valid]
    _stride = max(1, len(points3d) // 120000)
    points3d = points3d[::_stride].astype(np.float32, copy=False)
    colors = colors[::_stride].astype(np.float32, copy=False)
    return colors, points3d


@app.cell(hide_code=True)
def _(colors, go, point_density, points3d):
    _target = min(max(int(point_density.value), 1000), 120000)
    _plot_stride = max(1, len(points3d) // _target)
    _plot_points = points3d[::_plot_stride]
    _plot_colors = colors[::_plot_stride]
    _fig_cloud = go.Figure(
        data=[
            go.Scatter3d(
                x=_plot_points[:, 0],
                y=_plot_points[:, 1],
                z=_plot_points[:, 2],
                mode="markers",
                marker={"size": 1.2, "color": _plot_colors, "opacity": 0.7},
            )
        ]
    )
    _fig_cloud.update_layout(
        title="VGGT Point Cloud Visualization",
        scene={
            "xaxis_title": "X",
            "yaxis_title": "Y",
            "zaxis_title": "Z",
            "dragmode": "orbit",
        },
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
        height=760,
        uirevision="keep",
    )
    _fig_cloud
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## VGGT-Omega

    - CVPR 2026 Oral (Submitted May 14 2026 on Arxiv)

    - Recent paper published last week with several optimizations on top of VGGT
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
