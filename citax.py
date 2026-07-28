"""City-X synthetic benchmark: instance generation and cost components."""
import numpy as np

GRID_W, GRID_H = 48.0, 52.0
N_DISTRICTS = 5
N_SUBJECTS = 8
DISTRICT_GROWTH = np.array([-0.146, -0.146, 0.372, 0.372, 0.372])

SCENARIOS = [
    ("DOEM (pure distance)", 1.0, 0.0, 0.0),
    ("High physical cost", 0.8, 0.1, 0.1),
    ("Balanced", 0.4, 0.3, 0.3),
    ("High humanistic", 0.1, 0.1, 0.8),
    ("Pure humanistic", 0.0, 0.0, 1.0),
]


class Instance:
    def __init__(self, n_teachers, n_schools, school_xy, school_district,
                 school_subject, vacancies, teacher_xy, teacher_district,
                 teacher_subject, seed):
        self.n_teachers = n_teachers
        self.n_schools = n_schools
        self.school_xy = school_xy
        self.school_district = school_district
        self.school_subject = school_subject
        self.vacancies = vacancies
        self.teacher_xy = teacher_xy
        self.teacher_district = teacher_district
        self.teacher_subject = teacher_subject
        self.seed = seed

    @property
    def total_vacancies(self):
        return int(self.vacancies.sum())


def neyman_scott(rng, n_target, kappa=0.012, sigma=3.5):
    """Neyman-Scott cluster process; kappa is parent intensity per km^2."""
    n_parents = max(3, rng.poisson(kappa * GRID_W * GRID_H))
    parents = np.column_stack([rng.uniform(0, GRID_W, n_parents),
                               rng.uniform(0, GRID_H, n_parents)])
    per_parent = int(np.ceil(n_target / n_parents)) + 3
    pts = []
    for px, py in parents:
        k = rng.poisson(per_parent)
        if k:
            pts.append(np.column_stack([rng.normal(px, sigma, k),
                                        rng.normal(py, sigma, k)]))
    pts = np.vstack(pts)
    pts[:, 0] = np.clip(pts[:, 0], 0, GRID_W)
    pts[:, 1] = np.clip(pts[:, 1], 0, GRID_H)
    idx = rng.permutation(len(pts))[:n_target]
    if len(idx) < n_target:
        extra = rng.uniform([0, 0], [GRID_W, GRID_H], (n_target - len(idx), 2))
        return np.vstack([pts[idx], extra])
    return pts[idx]


def assign_district(xy):
    """Nearest-centre partition approximating administrative districts."""
    centers = np.array([
        [GRID_W * 0.35, GRID_H * 0.45], [GRID_W * 0.50, GRID_H * 0.55],
        [GRID_W * 0.20, GRID_H * 0.82], [GRID_W * 0.80, GRID_H * 0.25],
        [GRID_W * 0.82, GRID_H * 0.78],
    ])
    return np.linalg.norm(xy[:, None, :] - centers[None, :, :], axis=2).argmin(axis=1)


def make_instance(seed, n_teachers=5236, n_schools=114):
    rng = np.random.default_rng(seed)
    school_xy = neyman_scott(rng, n_schools)
    school_district = assign_district(school_xy)
    school_subject = rng.integers(0, N_SUBJECTS, n_schools)

    lam = np.clip(9.0 + 60.0 * DISTRICT_GROWTH[school_district], 1.0, None)
    vacancies = rng.poisson(lam) + 1

    teacher_xy = neyman_scott(rng, n_teachers, kappa=0.02, sigma=5.0)
    teacher_district = assign_district(teacher_xy)
    teacher_subject = rng.integers(0, N_SUBJECTS, n_teachers)

    return Instance(n_teachers, n_schools, school_xy, school_district,
                    school_subject, vacancies, teacher_xy, teacher_district,
                    teacher_subject, seed)


def build_cost_components(inst, rng, commute_kmh=25.0):
    """Return (distance, commute_minutes, Q, M_hat, R, I) as dense n-by-m arrays.

    Q is the statutory qualification matrix (adjacent subjects admissible).
    M_hat is normalised pedagogical misalignment within Q.
    R is the ordinal resistance label on {0, 1, 5, 10}.
    I flags cross-district pairs.
    """
    d = np.linalg.norm(inst.teacher_xy[:, None, :] - inst.school_xy[None, :, :],
                       axis=2)
    commute_min = d / commute_kmh * 60.0

    subj_gap = np.abs(inst.teacher_subject[:, None] - inst.school_subject[None, :])
    Q = subj_gap <= 1
    M_hat = subj_gap / (N_SUBJECTS - 1)
    I = (inst.teacher_district[:, None]
         != inst.school_district[None, :]).astype(np.float64)

    z = 0.030 * commute_min + 0.55 * I + rng.normal(0, 0.45, d.shape)
    R = np.select([z < 0.65, z < 1.35, z < 2.10], [0.0, 1.0, 5.0], default=10.0)
    return d, commute_min, Q, M_hat, R, I


def utility(R, k=0.2):
    """Exponential utility transform; k -> 0 recovers linear normalisation."""
    if k < 1e-9:
        return R / 10.0
    return (np.exp(k * R) - 1.0) / (np.exp(10.0 * k) - 1.0)
