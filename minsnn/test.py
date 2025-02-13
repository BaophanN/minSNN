




def save_checkpoint(epoch, iteration, model, optimizer, loss_hist, test_loss_hist):
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch{epoch}_iter{iteration}.pth')
    torch.save({
        'epoch': epoch,
        'iteration': iteration,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss_hist': loss_hist,
        'test_loss_hist': test_loss_hist
    }, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")

def plot_loss(loss_hist, test_loss_hist):
    plt.figure(figsize=(10, 5))
    plt.plot(loss_hist, label='Train Loss')
    plt.plot(test_loss_hist, label='Test Loss')
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Testing Loss Over Time")
    plt.show()

plot_loss(loss_hist, test_loss_hist)