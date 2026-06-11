"""
Minimal depth dropout + contrastive implementation
Add this to finetune_depthvla.py
"""

# ============================================================================
# Add to training loop (after depth_encoder forward)
# ============================================================================

# Depth dropout (simple version, no curriculum)
if self.training and self.config.depth_dropout > 0:
    dropout_mask = torch.rand(depth_features.size(0), device=depth_features.device)
    dropout_mask = (dropout_mask > self.config.depth_dropout).float().view(-1, 1, 1)
    depth_features = depth_features * dropout_mask

# ============================================================================
# Add to loss computation
# ============================================================================

# Main action loss
action_pred = self.action_head(rgb_context, depth_features, proprio)
loss_action = F.smooth_l1_loss(action_pred, action_labels)

# Contrastive loss (normal vs null)
if self.training and self.config.use_contrastive:
    # Forward with null depth
    depth_features_null = torch.zeros_like(depth_features)
    action_pred_null = self.action_head(rgb_context, depth_features_null, proprio)
    loss_action_null = F.smooth_l1_loss(action_pred_null, action_labels)

    # Ranking loss: normal must beat null
    margin = self.config.contrastive_margin  # 0.05
    loss_contrastive = F.relu(loss_action - loss_action_null + margin)

    # Total loss
    loss_total = loss_action + self.config.contrastive_weight * loss_contrastive

    # Logging
    metrics = {
        'loss/action': loss_action.item(),
        'loss/action_null': loss_action_null.item(),
        'loss/contrastive': loss_contrastive.item(),
        'loss/total': loss_total.item(),
    }
else:
    loss_total = loss_action
    metrics = {'loss/action': loss_action.item()}

# ============================================================================
# Add to config
# ============================================================================

# In argparse or config dict
parser.add_argument('--depth_dropout', type=float, default=0.3)
parser.add_argument('--use_contrastive', action='store_true', default=True)
parser.add_argument('--contrastive_weight', type=float, default=0.3)
parser.add_argument('--contrastive_margin', type=float, default=0.05)

