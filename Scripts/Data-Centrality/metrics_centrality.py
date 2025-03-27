import os
import pandas
import numpy as np
import pathlib
import csv
import statsmodels.api as sm
from scipy.linalg import eigh, pinv, LinAlgError

def LiuCentralityFunction(A):
    # Compute degrees (count non-zero elements in each row)
    d = np.count_nonzero(A, axis=1)  # Shape (N,)

    # Compute W using broadcasting
    W = d[:, None] + d[None, :]  # Outer sum of degrees
    # Set values to zero where A is zero
    W[A == 0] = 0
    return W


def YinLayerSimilarity(snapshot):
    N = snapshot.shape[0]
    T = snapshot.shape[2]
    NT = N*T

    # Compute C[j, t] for all j and t
    C_j_t = np.sum(snapshot[:, :, :-1] * snapshot[:, :, 1:], axis=0)  # Shape (N, T-1)
    # Initialize the big NT x NT matrix
    inter_layer_similarity = np.zeros((NT, NT))

    # Place C_j_t at off-diagonal (t, t+1) and (t+1, t) positions
    for t in range(T - 1):
        start_t = t * N
        start_t1 = (t + 1) * N

        inter_layer_similarity[start_t:start_t + N, start_t1:start_t1 + N] = np.diag(C_j_t[:, t])  # t to t+1
        inter_layer_similarity[start_t1:start_t1 + N, start_t:start_t + N] = np.diag(C_j_t[:, t])  # t+1 to t

    return inter_layer_similarity


def LiuLayerSimilarity(snapshot):
    N = snapshot.shape[0]
    T = snapshot.shape[2]
    NT = N*T
    # Compute numerator: sum over j of A[i,j,t] * A[i,j,t+1]
    numerator = np.sum(snapshot[:, :, :-1] * snapshot[:, :, 1:], axis=1)  # Shape (N, T-1)

    # Compute denominators separately
    denominator_t1 = np.sum(snapshot[:, :, 1:], axis=1)  # Sum over j for A[i,j,t+1] (Shape: N, T-1)
    denominator_t = np.sum(snapshot[:, :, :-1], axis=1)  # Sum over j for A[i,j,t] (Shape: N, T-1)

    # Avoid division by zero
    denominator_t1 = np.where(denominator_t1 == 0, 1e-10, denominator_t1)
    denominator_t = np.where(denominator_t == 0, 1e-10, denominator_t)

    # Compute S matrices
    S_t_t1 = numerator / denominator_t1  # S_{i,t,t+1} uses denominator of t+1
    S_t1_t = numerator / denominator_t  # S_{i,t+1,t} uses denominator of t

    # Initialize the big NT x NT matrix
    inter_layer_similarity = np.zeros((NT, NT))

    # Place S_t_t1 at (t, t+1) and S_t1_t at (t+1, t)
    for t in range(T - 1):
        start_t = t * N
        start_t1 = (t + 1) * N

        inter_layer_similarity[start_t:start_t + N, start_t1:start_t1 + N] = np.diag(S_t_t1[:, t])  # t → t+1
        inter_layer_similarity[start_t1:start_t1 + N, start_t:start_t + N] = np.diag(S_t1_t[:, t])  # t+1 → t
    return inter_layer_similarity


def HuangLayerSimilarity(snapshot):
    """
    Fits an ARMA (ARIMA with d=0) model to the degree sequence of each node from a series of adjacency matrices
    and constructs a block matrix where each block W[t, t-i] is a diagonal matrix containing
    the i-th coefficient of the ARMA model for each node at length t.

    Parameters:
    - adj_matrices: numpy.ndarray (NxNxT)
        A sequence of T adjacency matrices of size NxN.

    Returns:
    - inter_layer_similarity: numpy.ndarray
        The (N*T, N*T) block matrix with ARMA coefficients.
    """
    N, _, T = snapshot.shape  # Extract dimensions

    # Compute degree sequences for each node over time
    degree_sequences = np.sum(snapshot, axis=1)  # Shape: (N, T), summing over columns (node degrees)

    # Dictionary to store ARMA models for each node
    arma_models = {i: [] for i in range(N)}

    # Fit ARMA (via ARIMA with d=0) models for each node
    for i in range(N):  # Iterate over nodes
        for end_t in range(1, T + 1):  # From first step up to full sequence
            time_series = degree_sequences[i, :end_t]  # Extract sequence for node i up to end_t

            if len(time_series) > 1:  # ARMA needs at least two points
                model_order = (end_t - 1, 0, end_t - 1)  # (p, d, q) with d=0 (ARMA equivalent)
                try:
                    model = sm.tsa.ARIMA(time_series, order=model_order).fit()
                    arma_models[i].append(model)  # Store the model
                except Exception as e:
                    print(f"ARIMA (ARMA equivalent) failed for node {i} with sequence length {end_t}: {e}")
                    arma_models[i].append(None)  # Append None if model fitting fails

    # Initialize the (NT x NT) block matrix
    inter_layer_similarity = np.zeros((N * T, N * T))

    # Construct the block matrix using AR coefficients
    for t in range(1, T):  # Start from t=1 because t=0 has no previous coefficients
        for i in range(1, t + 1):  # Iterate over possible lags (t-i)
            start_t = t * N
            start_ti = (t - i) * N  # Previous time step block

            # Extract AR coefficients for time length t
            diag_entries = np.zeros(N)  # Default to zero if no coefficient available

            for node in range(N):
                if len(arma_models[node]) >= t and arma_models[node][t-1] is not None:
                    ar_coeffs = arma_models[node][t-1].arparams  # Extract AR coefficients
                    if len(ar_coeffs) >= i:
                        diag_entries[node] = ar_coeffs[i-1]  # Get the i-th coefficient

            # Fill diagonal block W[t, t-i] with the extracted coefficients
            inter_layer_similarity[start_t:start_t+N, start_ti:start_ti+N] = np.diag(diag_entries)

    return inter_layer_similarity

class SupraAdjacencyMatrix:

    def __init__(self, snapshot, inter_layer_similarity, centrality_function=None, epsilon=1.0):
        self._cc = None
        self._mlc = None
        self._mnc = None
        self._jc = None

        self._tac = None
        self._fom = None

        if centrality_function is None:
            centrality_function = lambda x: x
        N = snapshot.N
        T = snapshot.T
        NT = N*T
        self._orig_N = N
        self._orig_T = T
        self._orig_NT = NT
        snapshot = snapshot.tensor

        # Set centrality (adjacency) matrices to the SCM
        centrality_matrix = np.zeros((NT, NT))
        # Get indices for diagonal blocks
        idx = np.arange(T) * N
        # Assign values using NumPy advanced indexing
        for t in range(T):
            centrality_matrix[idx[t]:idx[t] + N, idx[t]:idx[t] + N] = epsilon*centrality_function(snapshot[:, :, t])

        # Set the inter-layer similarity to the SCM
        if inter_layer_similarity is None:
            inter_layer_similarity = np.zeros((NT, NT))
        elif isinstance(inter_layer_similarity, str):
            if inter_layer_similarity not in ["F", "B", "FB", "BF", "C"]:
                raise ValueError("time_coupling must be one of the following: F, B, BF, FB, C")
            if inter_layer_similarity == 'C':
                # Create the block matrix using Kronecker product with identity matrix
                inter_layer_similarity = np.triu(np.ones((T, T)), k=1)
            else:
                ils = np.zeros((T,T))
                if "F" in inter_layer_similarity:
                    ils += np.eye(T, k=1)
                if "B" in inter_layer_similarity:
                    ils += np.eye(T, k=-1)
                inter_layer_similarity = ils
        elif callable(inter_layer_similarity):
            inter_layer_similarity = inter_layer_similarity(snapshot)

        if isinstance(inter_layer_similarity, np.ndarray):
            if inter_layer_similarity.shape == (T, T):
                ils = np.kron(inter_layer_similarity, np.eye(N))
            elif inter_layer_similarity.shape == (NT, NT):
                ils = inter_layer_similarity
            else:
                raise ValueError(f"Cannot use time_coupling of shape {inter_layer_similarity.shape}; must be either (T,T) or (NT, NT)")
        else:
            raise ValueError("Time coupling must be a numpy.ndarray, callable or string (or None)")

        self._centrality_matrix = centrality_matrix
        self._inter_layer_similarity = inter_layer_similarity
        self._supra = centrality_matrix + ils

    def compute_centrality(self):
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(self._supra)

        # Find the index of the largest eigenvalue
        max_eigenvalue_index = np.argmax(eigenvalues)

        # Get the corresponding eigenvector
        dominant_eigenvector = eigenvectors[:, max_eigenvalue_index]
        # Ensure it has a positive first element (or flip signs)
        if dominant_eigenvector[0] < 0:
            dominant_eigenvector = -dominant_eigenvector
        self._jc = np.reshape(dominant_eigenvector, shape=(self._orig_T, self._orig_N)).T
        self._mnc = np.sum(self._jc, axis=1)
        self._mlc = np.sum(self._jc, axis=0)
        self._cc = self._jc / self._mlc


    @property
    def supracentrality_matrix(self):
        return self._supra


    @property
    def scm(self):
        return self._supra


    @property
    def joint_centrality(self):
        return self._jc

    @property
    def jc(self):
        return self._jc

    @property
    def marginal_node_centrality(self):
        return self._mnc

    @property
    def mnc(self):
        return self._mnc

    @property
    def marginal_layer_centrality(self):
        return self._mlc

    @property
    def mlc(self):
        return self._mlc

    @property
    def conditional_centrality(self):
        return self._cc

    @property
    def cc(self):
        return self._cc

    def zero_first_order_expansion(self, fom=True):
        """Compute the time-averaged centrality and first-order mover scores.
        """

        NT = self._orig_NT
        T = self._orig_T
        N = self._orig_N

        # Timed-averaged centralities (zeroth order expansion)
        # Step 1: Compute X^(1) using Eq. (4.14)
        A = self._inter_layer_similarity

        if A.shape != (T,T):
            raise LinAlgError("TAC and FOM scores can only be computed if inter_layer_similarity is (T,T) matrix")

        lambda0, u, X1 = self._eig_layer_similarity()

        U_matrix = np.zeros((NT, N))  # Store all u_i vectors
        for i in range(N):
            U_matrix[i * T:(i + 1) * T, i] = u  # Set u in the correct block

        if X1 is None:
            X1 = self._X1(U_matrix)

        # Step 2: Solve eigenvector equation X^(1) alpha = λ1 alpha
        eigenvalues, eigenvectors = eigh(X1)  # Compute all eigenvalues & eigenvectors
        lambda1 = eigenvalues[-1]  # Largest eigenvalue (last element)
        alpha = eigenvectors[:, -1]  # Corresponding eigenvector (last column)

        self._tac = alpha

        if not fom:
            return

        # First-order-mover scores (first order expansion)
        # Step 3: Compute X^(2) using Eq. (4.22)
        L0 = pinv(lambda0 * np.eye(T) - A)  # Compute (λ0 I - A)†
        L0_pinv = np.kron(L0, np.eye(N))  # Compute L_0 = (λ0 I - A)† ⊗ I


        G = self._centrality_matrix
        X2 = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                # Construct block vectors u_i and u_j
                u_i = U_matrix[:, i]  # Precomputed u_i
                u_j = U_matrix[:, j]  # Precomputed u_j

                X2[i, j] = u_i.T @ G @ L0_pinv @ G @ u_j

        # Step 4: Compute first-order correction beta
        lambda2 = alpha.T @ X2 @ alpha  # Compute λ2
        beta = np.linalg.solve(X1 - lambda1 * np.eye(N), (lambda2 * np.eye(N) - X2) @ alpha)

        # Step 5: Compute first-order mover scores
        v0 = U_matrix @ alpha  # Compute v0
        L0_G_v0 = L0_pinv @ G @ v0  # Compute L0† G v0
        mover_scores = np.sqrt(beta ** 2 + np.sum(L0_G_v0.reshape(N, T) ** 2, axis=1))  # Compute final scores

        self._fom = mover_scores

    def _eig_layer_similarity(self):
        eigenvalues_A, eigenvectors_A = eigh(self._inter_layer_similarity)
        u = eigenvectors_A[:, -1]  # Take the eigenvector corresponding to the largest eigenvalue
        u /= np.linalg.norm(u)  # Normalize
        lambda0 = eigenvalues_A[-1]
        return lambda0, u, None

    def _X1(self, U_matrix):
        N = self._orig_N
        X1 = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                X1[i, j] = U_matrix[:, i].T @ self._centrality_matrix @ U_matrix[:, j]
        return X1

    @property
    def tac(self):
        return self._tac

    @property
    def time_averaged_centrality(self):
        return self._tac

    @property
    def fom(self):
        return self._fom

    @property
    def first_order_mover_scores(self):
        return self._fom



class TaylorSupraMatrix(SupraAdjacencyMatrix):

    def __init__(self, snapshot, epsilon=1.0, centrality_function=None):
        super().__init__(snapshot, inter_layer_similarity='FB', centrality_function=centrality_function,
                         epsilon=epsilon)


    def _eig_layer_similarity(self):
        T = self._orig_T
        N = self._orig_N

        lambda0 = 2 * np.cos(np.pi / (T + 1))  # Directly from Eq. (4.4)
        X1 = np.zeros((N, N))
        sin_vector = []
        gamma1 = 0.0
        for t in range(T):
            sin_factor = np.sin(np.pi * (t + 1) / (T + 1))
            sin_vector.append(sin_factor)
            sin_factor = sin_factor ** 2
            gamma1 += sin_factor
            C_t = self._supra[t * N:(t + 1) * N, t * N:(t + 1) * N]  # Extract N×N block at time t
            X1 += C_t * sin_factor  # Apply the weighting

        X1 /= gamma1  # Normalize
        u = np.array(sin_vector) / np.sqrt(gamma1)
        return lambda0, u, X1


class YinSupraMatrix(SupraAdjacencyMatrix):

    def __init__(self, snapshot):
        super().__init__(snapshot, inter_layer_similarity=YinLayerSimilarity)


class LiuSupraMatrix(SupraAdjacencyMatrix):

    def __init__(self, snapshot):
        super().__init__(snapshot, inter_layer_similarity=LiuLayerSimilarity, centrality_function=LiuCentralityFunction)

class HuangSupraMatrix(SupraAdjacencyMatrix):

    def __init__(self, snapshot):
        super().__init__(snapshot, inter_layer_similarity=HuangLayerSimilarity)

class SnapshotGraph:

    def __init__(self):

        self._tensor = None
        self._vertices: list = []
        self._timestamps: list = []
        self._vertex_index_mapping: dict[str, int] = {}
        self._timestamp_index_mapping: dict[str, int] = {}

    @property
    def tensor(self):
        return self._tensor

    @property
    def vertices(self):
        return self._vertices

    @vertices.setter
    def vertices(self, new_value):
        assert type(new_value) is list
        assert len(self._vertices) == len(new_value)
        self._vertices = new_value
        self._vertex_index_mapping = {value: index for index, value in enumerate(new_value)}


    @property
    def timestamps(self):
        return self._timestamps

    @timestamps.setter
    def timestamps(self, new_value):
        assert type(new_value) is list
        assert len(self._timestamps) == len(new_value)
        self._timestamps = new_value
        self._timestamp_index_mapping = {value: index for index, value in enumerate(new_value)}

    @property
    def N(self):
        return len(self._vertices)

    @property
    def T(self):
        return len(self._timestamps)

    def load_csv(self, csv_file, /, *, source='source', target='target', timestamp='timestamp', weight='weight',
                       directed=True, dtype=np.float32, sort_vertices=False, sort_timestamps=False):
        self.load_csv_list([csv_file], source=source, target=target, timestamp=timestamp, weight=weight,
                           directed=directed, dtype=dtype, sort_timestamps=sort_timestamps, sort_vertices=sort_vertices)

    def load_edge_list(self, edge_list, vertex_list, timestamp_list,
                       directed=True, dtype=np.float32):
        vertex_index_mapping = {value: index for index, value in enumerate(vertex_list)}
        timestamp_index_mapping = {value: index for index, value in enumerate(timestamp_list)}
        max_vertex = len(vertex_list)
        max_time = len(timestamp_list)
        tensor = np.full((max_vertex, max_vertex, max_time), 0.0, dtype=dtype)
        for i, j, t, w in edge_list:
            i = vertex_index_mapping[i]
            j = vertex_index_mapping[j]
            t = timestamp_index_mapping[t]
            w = float(w)
            tensor[i, j, t] = w
            if directed is False:
                tensor[j, i, t] = w
        self._tensor = tensor
        self._vertices = vertex_list
        self._timestamps = timestamp_list
        self._vertex_index_mapping = vertex_index_mapping
        self._timestamp_index_mapping = timestamp_index_mapping


    def load_csv_list(self, csv_list, /, *, source='source', target='target',
                      timestamp='timestamp', weight='weight',
                      directed=True, dtype=np.float32, sort_vertices=False, sort_timestamps=False):

        rows = []
        vertex_set = set()
        timestamp_set = set()
        for input_file in csv_list:
            with open(input_file, 'r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    source_ = row[source]
                    target_ = row[target]
                    timestamp_ = str(pathlib.Path(input_file).with_suffix('')
                                     ) if timestamp is None else row[timestamp]
                    vertex_set.add(source_)
                    vertex_set.add(target_)
                    timestamp_set.add(timestamp_)
                    rows.append([source_, target_, timestamp_, row.get(weight, 1.0)])

        vertex_list = list(vertex_set)
        vertex_list.sort() if sort_vertices else None
        timestamp_list = list(timestamp_set)
        timestamp_list.sort() if sort_timestamps else None
        self.load_edge_list(rows, vertex_list, timestamp_list, directed, dtype)


# Load the complete TN with all the releases (version 1.0.0)
tempnet = SnapshotGraph()
tempnet.load_csv(os.path.join("raw_data", "temp_net", "train-ticket-temporal-v1.0.0.csv"),
                 source="source", target="target", timestamp="version",
                 weight="weight", sort_timestamps=True, sort_vertices=True)

temporal_rows = [["MS_system", "Microservice",
         "Taylor_JC", "Yin_JC", "Liu_JC", "Huang_JC",
         "Taylor_CC", "Yin_CC", "Liu_CC", "Huang_CC",
         "Taylor_TAC", "Taylor_FOM", "Taylor_FOM_NORM"
                  ]]

static_rows = [["Microservice",
                "Taylor_MNC", "Yin_MNC", "Liu_MNC", "Huang_MNC",
                ]]

time_rows = [["Version", "Version Id",
              "Taylor_MLC", "Yin_MLC", "Liu_MLC", "Huang_MLC",
              ]]

versions = tempnet.timestamps

# Apply Taylor algorithm
taylor = TaylorSupraMatrix(tempnet)
taylor.compute_centrality()
taylor_joint = taylor.joint_centrality
taylor_cc = taylor.cc
taylor_mnc = taylor.mnc
taylor_mlc = taylor.mlc

# Apply Liu algorithm
liu = LiuSupraMatrix(tempnet)
liu.compute_centrality()
liu_joint = liu.joint_centrality
liu_cc = liu.cc
liu_mnc = liu.mnc
liu_mlc = liu.mlc

# Apply Huang algorithm
huang = HuangSupraMatrix(tempnet)
huang.compute_centrality()
huang_joint = huang.joint_centrality
huang_cc = huang.cc
huang_mnc = huang.mnc
huang_mlc = huang.mlc

# Apply Huang algorithm
yin = YinSupraMatrix(tempnet)
yin.compute_centrality()
yin_joint = yin.joint_centrality
yin_cc = yin.cc
yin_mnc = yin.mnc
yin_mlc = yin.mlc

service_mapping_latest = tempnet._vertex_index_mapping


for version_id, version in enumerate(versions):
    # Load the cumulative TN for each version
    tempnet = SnapshotGraph()
    tempnet.load_csv(os.path.join("raw_data", "temp_net", f"train-ticket-temporal-{version}.csv"),
                     source="source", target="target", timestamp="version",
                     weight="weight", sort_timestamps=True, sort_vertices=True)
    taylor = TaylorSupraMatrix(tempnet)
    taylor.compute_centrality()
    # Solve for Taylor TAC, FOM for the cumulative TN
    taylor.zero_first_order_expansion()
    taylor_tac = taylor.tac
    taylor_fom = taylor.fom
    # Normalize FOM by L2 norm
    fom_norm = np.linalg.norm(taylor_fom)
    taylor_fom_norm = taylor_fom / fom_norm

    service_mapping_current = tempnet._vertex_index_mapping

    # Write data to csv rows
    time_rows.append([version, version_id+1,
                      # Marginal Layer Centralities
                      abs(float(taylor_mlc[version_id])),
                      abs(float(yin_mlc[version_id])),
                      abs(float(liu_mlc[version_id])),
                      abs(float(huang_mlc[version_id])),
                      ])
    # Write data for each service in the complete TN
    for service, service_id in service_mapping_latest.items():

        new_row = [f"train-ticket-{version[1:]}", service,
               # Joint Centralities
               abs(float(taylor_joint[service_id, version_id])),
               abs(float(yin_joint[service_id, version_id])),
               abs(float(liu_joint[service_id, version_id])),
               abs(float(huang_joint[service_id, version_id])),
               # Conditional Centralities
               abs(float(taylor_cc[service_id, version_id])),
               abs(float(yin_cc[service_id, version_id])),
               abs(float(liu_cc[service_id, version_id])),
               abs(float(huang_cc[service_id, version_id])),
               ]

        # Write TAC/FOM if the service is found in current cumulative TN
        if service in service_mapping_current:
            service_id_current = service_mapping_current[service]
            new_row.extend([
                # Time-averaged centralities
                taylor_tac[service_id_current],
                # First-order-mover scores
                taylor_fom[service_id_current],
                taylor_fom_norm[service_id_current],
            ])
        else:
            new_row.extend([0.0, 0.0, 0.0])
        temporal_rows.append(new_row)

        # Write MNC only once
        if version_id == 6:
            static_rows.append([service,
                              # Marginal Node Centralities
                              abs(float(taylor_mnc[service_id])),
                              abs(float(yin_mnc[service_id])),
                              abs(float(liu_mnc[service_id])),
                              abs(float(huang_mnc[service_id])),
                              ])

df = pandas.DataFrame(temporal_rows)
df.to_csv("Metrics/metrics_temporal_centrality.csv", index=False, header=False)
df = pandas.DataFrame(static_rows)
df.to_csv("Metrics/metrics_mnc.csv", index=False, header=False)
df = pandas.DataFrame(time_rows)
df.to_csv("Metrics/metrics_mlc.csv", index=False, header=False)
