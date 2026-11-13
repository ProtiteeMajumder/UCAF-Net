import torch
from loss_metrics import uncertainty_loss, compute_metrics

# ============================================================
# TRAIN FUNCTION (UNCERTAINTY-AWARE)
# ============================================================

def train_one_epoch(model, loader, optimizer, scheduler, device):

    model.train()

    all_preds, all_labels = [], []

    running_loss = 0.0

    for images, labels in loader:

        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        # ----------------------------------------------------
        # FORWARD PASS
        # ----------------------------------------------------

        outputs, sigma_r, sigma_m = model(images)

        # ----------------------------------------------------
        # UNCERTAINTY LOSS
        # ----------------------------------------------------

        loss = uncertainty_loss(
            outputs,
            labels,
            sigma_r,
            sigma_m
        )

        # ----------------------------------------------------
        # BACKPROPAGATION
        # ----------------------------------------------------

        loss.backward()

        optimizer.step()

        # ----------------------------------------------------
        # LR SCHEDULER STEP
        # ----------------------------------------------------

        scheduler.step()

        running_loss += loss.item() * images.size(0)

        # ----------------------------------------------------
        # PREDICTIONS
        # ----------------------------------------------------

        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.cpu().numpy())

        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)

    acc, prec, rec, f1 = compute_metrics(
        all_labels,
        all_preds
    )

    return epoch_loss, acc, prec, rec, f1

# ============================================================
# VALIDATION FUNCTION (UNCERTAINTY-AWARE)
# ============================================================

def validate_one_epoch(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    running_loss = 0.0

    with torch.no_grad():
        for images, labels in loader:   

            images, labels = images.to(device), labels.to(device)

            outputs, sigma_r, sigma_m = model(images)

            loss = uncertainty_loss(outputs, labels, sigma_r, sigma_m)

            running_loss += loss.item() * images.size(0)

            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)

    acc, prec, rec, f1 = compute_metrics(all_labels, all_preds)

    return epoch_loss, acc, prec, rec, f1

# ============================================================
# TEST FUNCTION (UNCERTAINTY-AWARE)
# ============================================================

def test_model(model, loader, device, class_names=None):
    model.eval()
    all_preds, all_labels = [], []
    running_loss = 0.0

    with torch.no_grad():
        for images, labels in loader:   

            images, labels = images.to(device), labels.to(device)

            outputs, sigma_r, sigma_m = model(images)

            loss = uncertainty_loss(outputs, labels, sigma_r, sigma_m)

            running_loss += loss.item() * images.size(0)

            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = running_loss / len(loader.dataset)

    acc, prec, rec, f1 = compute_metrics(all_labels, all_preds)

    return avg_loss, acc, prec, rec, f1, all_labels, all_preds

# ============================================================
# FULL TRAINING LOOP (UNCERTAINTY-AWARE)
# ============================================================

def run_training(model,
                 train_loader,
                 val_loader,
                 test_loader,
                 optimizer,
                 scheduler,
                 device,
                 class_names=None,
                 num_epochs=200):

    best_val_loss = float('inf')

    # --------------------------------------------------------
    # HISTORY
    # --------------------------------------------------------
    train_losses = []
    val_losses = []

    train_accuracies = []
    val_accuracies = []

    for epoch in range(num_epochs):

        # ----------------------------------------------------
        # TRAINING
        # ----------------------------------------------------
        train_loss, train_acc, train_prec, train_rec, train_f1 = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            device
        )

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------
        val_loss, val_acc, val_prec, val_rec, val_f1 = validate_one_epoch(
            model,
            val_loader,
            device
        )

        # ----------------------------------------------------
        # STORE HISTORY
        # ----------------------------------------------------
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        print(f"\nEpoch [{epoch+1}/{num_epochs}]")

        print(f"Train - Loss: {train_loss:.4f}, "
              f"Acc: {train_acc:.4f}, "
              f"Prec: {train_prec:.4f}, "
              f"Recall: {train_rec:.4f}, "
              f"F1: {train_f1:.4f}")

        print(f"Val   - Loss: {val_loss:.4f}, "
              f"Acc: {val_acc:.4f}, "
              f"Prec: {val_prec:.4f}, "
              f"Recall: {val_rec:.4f}, "
              f"F1: {val_f1:.4f}")

        # ----------------------------------------------------
        # SAVE BEST MODEL
        # ----------------------------------------------------
        if val_loss < best_val_loss:

            best_val_loss = val_loss

            torch.save(
                model.state_dict(),
                'best_model.pth'
            )

            print(f"✅ Saved best model "
                  f"(val_loss: {val_loss:.4f})")

    # --------------------------------------------------------
    # LOAD BEST MODEL
    # --------------------------------------------------------
    model.load_state_dict(
        torch.load('best_model.pth')
    )

    print("📥 Loaded the best saved model for final testing.")

    # --------------------------------------------------------
    # TESTING
    # --------------------------------------------------------
    print("\n🔍 Evaluating on Test Set:")

    test_loss, test_acc, test_prec, test_rec, test_f1, all_labels, all_preds = test_model(
        model,
        test_loader,
        device,
        class_names
    )

    print(f"Test - Loss: {test_loss:.4f}, "
          f"Acc: {test_acc:.4f}, "
          f"Prec: {test_prec:.4f}, "
          f"Recall: {test_rec:.4f}, "
          f"F1: {test_f1:.4f}")

    # --------------------------------------------------------
    # FINAL ACCURACY
    # --------------------------------------------------------
    corrects = torch.eq(
        torch.tensor(all_labels),
        torch.tensor(all_preds)
    )

    num_correct = corrects.sum().item()

    total = len(corrects)

    print(f"✅ Correct Predictions: "
          f"{num_correct}/{total} "
          f"({100 * num_correct / total:.2f}%)")

    # --------------------------------------------------------
    # RETURN HISTORY
    # --------------------------------------------------------
    history = {
        "train_loss": train_losses,
        "val_loss": val_losses,
        "train_acc": train_accuracies,
        "val_acc": val_accuracies,
    }

    return history
