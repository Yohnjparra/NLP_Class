# Latent Space, VAEs & GANs — Lecture + Assignment on NRP JupyterHub

This module teaches **latent spaces** through two generative models, **Variational Autoencoders (VAEs)** and **Generative Adversarial Networks (GANs)**. It runs on the **National Research Platform (NRP) Nautilus JupyterHub**. Everything is plain PyTorch and torchvision, and the datasets download automatically.

## What's in this folder

| File | Who | What |
|---|---|---|
| `01_Lecture_Latent_Space_VAE_GAN.ipynb` | Everyone | Lecture notebook covering theory, a VAE and a GAN on MNIST, and latent-space visualisation |
| `02_Assignment_Latent_Space.ipynb` | Students | Assignment on Fashion-MNIST: 9 coding TODOs with auto-checks, 6 written questions, and a bonus |
| `README.md` | Everyone | This file: how to run everything on NRP |

### Learning objectives
1. Explain what a latent space is (compact, continuous, complete).
2. Explain the VAE objective (reconstruction + KL) and the reparameterization trick.
3. Explain the GAN minimax game and the non-saturating generator loss.
4. Visualise and navigate latent spaces through scatter plots, decoded manifolds, interpolation (lerp and slerp), and latent arithmetic.
5. Compare VAEs and GANs: sharpness, mode coverage, stability, and whether the model has an encoder.

---

## 1. Get access to NRP (one time)

1. Go to the **NRP portal** at <https://nrp.ai> and click **Login**. Sign in through **CILogon** with your **university credentials** (choose your institution from the list).
2. The NRP JupyterHub needs no namespace or Kubernetes setup. Logging in with your institutional account is enough for most universities.
   - If your institution isn't listed, or the login is refused, ask your instructor. They can request access through the NRP support Matrix/Element chat linked from the portal.

> 📘 Official docs: <https://nrp.ai/documentation/userdocs/jupyter/jupyterhub-service/>. If any URL below has changed, the docs have the current one.

## 2. Start a GPU Jupyter server

1. Open the NRP JupyterHub at **<https://jupyterhub-west.nrp-nautilus.io>** (it's also linked from the NRP portal under *JupyterHub*).
2. Click **Sign in with CILogon** and pick your institution.
3. On the **Server Options** page, choose:

   | Setting | Recommended value | Notes |
   |---|---|---|
   | GPUs | **1** | Any GPU type is fine. These models are small. |
   | CPU cores | 4 | |
   | Memory (RAM) | 16 GB | 8 GB also works |
   | Image | A **PyTorch / data-science stack** image | Anything that includes `torch`, `torchvision` and `matplotlib` |

4. Click **Start**. The first start can take **2–5 minutes** while the image is pulled to a node. If it waits a long time at "Pending", the GPUs you asked for may all be in use. See [Troubleshooting](#5-troubleshooting).

## 3. Get the notebooks onto your server

**Option A: Clone with Git (recommended)**
1. In JupyterLab, open **File → New → Terminal**.
2. Run:
   ```bash
   git clone https://github.com/Yohnjparra/NLP_Class.git
   ```
3. In the file browser on the left, double-click the new `NLP_Class` folder to open it.

If your instructor updates the materials later, get the changes from that folder in the terminal:
```bash
cd ~/NLP_Class && git pull
```
> ⚠️ Do the assignment in a **copy** of the notebook (right-click → **Duplicate**), not the original. Otherwise `git pull` can conflict with your edits.

**Option B: Upload**
1. On the GitHub page, click **Code → Download ZIP** and unzip it on your computer.
2. In JupyterLab, click the **Upload** button (⬆ arrow at the top of the file browser) and select the two `.ipynb` files.

Your home directory (`/home/jovyan`) is **persistent storage**, so files and the downloaded datasets (saved to `~/data`) survive server restarts.

## 4. Run the notebooks

1. **Check the GPU.** In a terminal, run:
   ```bash
   nvidia-smi
   ```
   You should see one GPU listed. The first code cell of each notebook also prints `device: cuda` and the GPU name.
2. **Check the packages** (optional). Most PyTorch images already have these. If one is missing:
   ```bash
   pip install --user torch torchvision matplotlib
   ```
   Then use **Kernel → Restart Kernel**.
3. Open **`01_Lecture_Latent_Space_VAE_GAN.ipynb`** and run the cells top to bottom (**Shift + Enter**, or **Run → Run All Cells**).
4. Then open **`02_Assignment_Latent_Space.ipynb`** and complete the TODOs.

### Expected run times

| Notebook | GPU (1× any NRP GPU) | CPU only |
|---|---|---|
| Lecture (VAE 15 epochs + GAN 30 epochs) | ~5 min | ~25–40 min |
| Assignment (2 VAEs, 3 β-VAEs, GAN, classifier) | ~12 min | ~60+ min |

On CPU, lower the `VAE_EPOCHS` / `GAN_EPOCHS` constants (for example to 5 and 10). Results will be rougher but still show every concept.

### Submitting the assignment
1. **Kernel → Restart Kernel and Run All Cells.** Make sure every sanity check prints ✅.
2. **File → Download** to get the `.ipynb` file.
3. **File → Save and Export Notebook As → HTML** to get a read-only copy with all plots.
4. Upload both files to your course LMS.

### When you're done: stop your server
**File → Hub Control Panel → Stop My Server.** GPUs are a shared resource on NRP, so please don't leave idle servers running. Idle servers are also shut down automatically after a period of inactivity. Your files in the home directory are kept, but **unsaved notebook state and variables are lost**, so save often.

---

## 5. Troubleshooting

| Symptom | Fix |
|---|---|
| Server stuck on **Pending** for more than 10 minutes | The GPUs you asked for are all busy. Stop the server, then restart it with a different GPU type, or with 0 GPUs (CPU mode) and fewer epochs. |
| `device: cpu` even though you asked for a GPU | Check that you selected **1 GPU** in Server Options and that the image includes CUDA-enabled PyTorch. Run `nvidia-smi` in a terminal. |
| `RuntimeError: DataLoader worker ... bus error` / shared-memory error | The notebooks already use `num_workers=0`. Keep it that way, because pods have a small `/dev/shm`. |
| Dataset download fails / HTTP error | Re-run the cell, since the torchvision mirrors are sometimes flaky. If it keeps failing, delete `~/data/MNIST` or `~/data/FashionMNIST` and try again. |
| `CUDA out of memory` | Another notebook may be holding the GPU. **Kernel → Shut Down All Kernels**, then rerun. You can also reduce `batch_size`. |
| `ModuleNotFoundError` | Run `pip install --user <package>` in a terminal, then restart the kernel. |
| Kernel dies / restarts during training | Usually out of RAM. Restart the server with 16 GB of memory. |
| GAN samples are noise or all look the same | GANs are noisy. Re-run the training cell (new random seed), or train for more epochs. Mode collapse is something to *discuss* in Q3.5, not a bug. |

---

## 6. Instructor notes

**Suggested lecture flow (about 80 min).** Train the models live while you lecture. Start the VAE training cell, explain ELBO and reparameterization while it runs, then do the same with the GAN.

| Time | Section in notebook |
|---|---|
| 0–10 | Part 0–1: setup, manifold hypothesis, what makes a "good" latent space |
| 10–30 | Part 2: VAE theory (ELBO, KL closed form, reparameterization), start training |
| 30–45 | Part 3: latent scatter, decoded manifold, interpolation, sampling. Point out the blurriness. |
| 45–65 | Part 4: GAN theory, non-saturating loss, live training, loss curves |
| 65–75 | Part 5: slerp vs. lerp, single-dimension walks (entanglement) |
| 75–80 | Part 6: comparison table and discussion questions |

**Assignment grading (100 pts + 10 bonus)**

| Part | Item | Pts | Auto-checked? |
|---|---|---|---|
| 1 | TODO 1.1 reparameterize | 5 | ✅ |
| 1 | TODO 1.2 VAE loss | 10 | ✅ |
| 1 | Q1.4 d=2 vs d=16 | 5 | |
| 1 | TODO 1.5 prior samples | 5 | ✅ (shape/range) |
| 2 | Q2.1 cluster analysis | 4 | |
| 2 | TODO 2.2 PCA | 6 | ✅ |
| 2 | TODO 2.3 interpolation + Q2.3 | 8 + 4 | ✅ (code) |
| 2 | TODO 2.4 centroids / arithmetic | 6 | ✅ |
| 2 | Q2.5 β trade-off | 7 | |
| 3 | TODO 3.1 / 3.2 GAN steps | 5 + 5 | ✅ |
| 3 | TODO 3.4 slerp interpolation | 5 | visual |
| 3 | Q3.5 sharpness vs. coverage | 5 | |
| 4 | Written reflection 4.1–4.3 | 15 | |
| Bonus | DCGAN | +10 | |

