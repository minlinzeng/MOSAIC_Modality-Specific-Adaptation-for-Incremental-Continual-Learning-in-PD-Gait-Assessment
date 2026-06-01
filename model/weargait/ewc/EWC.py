import torch
from torch import nn, optim, autograd
from torch.utils.data import DataLoader
import torch.nn.functional as F


class ElasticWeightConsolidation:
    """
    Implementation of Elastic Weight Consolidation (EWC).
    After finishing each task:
      1. Estimate diagonal Fisher information from gradients.
      2. Save current parameter values as task anchors.
    During later tasks:
      - Adds Fisher-weighted quadratic penalty to prevent important weights from drifting.
    """

    def __init__(self, model, criterion, lr=1e-3, weight=1000.0, weight_decay=0.0):
        self.model = model
        self.criterion = criterion
        self.weight = weight                   # λ: EWC regularization strength
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.task_history = []

    # ------------------------------------------------------------
    # Step 1: snapshot current optimal parameters (anchors of the Fisher) for new task
    # ------------------------------------------------------------
    def _snapshot_params(self, task_id: int):
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            buffer_name = name.replace('.', '__') + f'_anchor_t{task_id}'
            if hasattr(self.model, buffer_name):
                getattr(self.model, buffer_name).data.copy_(p.data)
            else:
                self.model.register_buffer(buffer_name, p.data.clone())

    # ------------------------------------------------------------
    # Step 2: compute and record Fisher information (diagonal)
    # ------------------------------------------------------------
    def _update_fisher_diag(self, dataset, task_id, batch_size=64, num_batches=50):
        """
        Calculates the diagonal of the Fisher Information Matrix using a Per-Sample loop.
        """
        self.model.eval()
        device = next(self.model.parameters()).device

        # 1. Handle DataLoader vs Dataset input
        if isinstance(dataset, DataLoader):
            dataloader = dataset
        else:
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

        # 2. Initialize Fisher accumulators (start at zero)
        # We filter for requires_grad to skip frozen parameters immediately
        names, params = zip(*[(n, p) for n, p in self.model.named_parameters() if p.requires_grad])
        fisher = {n: torch.zeros_like(p, device=device) for n, p in zip(names, params)}
        n_samples_processed = 0
        
        # 3. Outer Loop: Iterate over Batches
        for i, (x, y) in enumerate(dataloader):
            if i >= num_batches:
                break
            
            x, y = x.to(device), y.to(device)
            current_batch_size = x.size(0)
            # 4. Forward Pass (Once per batch for efficiency)
            log_probs = torch.log_softmax(self.model(x), dim=1)

            # 5. Inner Loop: Iterate over Samples
            for j in range(current_batch_size):
                self.model.zero_grad(set_to_none=True)
                log_prob_true = log_probs[j, y[j]]
                grads = torch.autograd.grad(
                    log_prob_true, 
                    params, 
                    retain_graph=True if j < current_batch_size - 1 else False,
                    create_graph=False,
                    allow_unused=True
                )
                for n, g in zip(names, grads):
                    if g is not None:
                        fisher[n] += g.detach() ** 2
                n_samples_processed += 1
        # 6. Normalize: Average across total samples seen
        for n in fisher:
            fisher[n] /= max(1, n_samples_processed)

        # 7. Store or Register Buffer
        for n, p in zip(names, params):
            buffer_name = n.replace('.', '__') + f'_fisher_diag_t{task_id}'
            if hasattr(self.model, buffer_name):
                getattr(self.model, buffer_name).data.copy_(fisher[n])
            else:
                self.model.register_buffer(buffer_name, fisher[n].clone())

            
    # ------------------------------------------------------------
    # Step 3: public API to register Fisher + anchors after a task
    # ------------------------------------------------------------
    def register_ewc_params(self, dataset, task_id: int, batch_size=64, num_batches=50):
        self._update_fisher_diag(dataset, task_id, batch_size, num_batches)
        self._snapshot_params(task_id)
        if task_id not in self.task_history:
            self.task_history.append(task_id)

        
    # ------------------------------------------------------------
    # Step 4: compute the EWC penalty term
    # ------------------------------------------------------------
    def _ewc_penalty(self):
        device = next(self.model.parameters()).device
        total = torch.zeros((), device=device)
        if not getattr(self, "task_history", None):
            return 0.5 * self.weight * total # no prior tasks → zero penalty as a tensor

        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            prefix = name.replace('.', '__')
            
            for t in self.task_history:
                anchor = getattr(self.model, f'{prefix}_anchor_t{t}', None)
                fisher = getattr(self.model, f'{prefix}_fisher_diag_t{t}', None)
                if anchor is None or fisher is None:
                    continue
                total = total + (fisher * (p - anchor) ** 2).sum()
        return 0.5 * self.weight * total

    # ------------------------------------------------------------
    # Step 5: normal training step with EWC regularization
    # ------------------------------------------------------------
    def forward_backward_update(self, x, y):
        self.model.train()
        self.optimizer.zero_grad()

        output = self.model(x)
        task_loss = self.criterion(output, y)
        ewc_loss = self._ewc_penalty()
        loss = task_loss + ewc_loss

        loss.backward()
        self.optimizer.step()
        return loss.item(), task_loss.item(), ewc_loss.item(), output

    # ------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------
    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path, map_location=None):
        self.model.load_state_dict(torch.load(path, map_location=map_location))
