import torch



# Function to train the model
def train_model(model, train_loader, criterion, optimizer, num_epochs=5, tasks_flattened=False):
    loss_canonical = []
    for epoch in range(num_epochs):
        print('epoch', epoch)
        model.train()
        for inputs, labels in train_loader:
            if tasks_flattened is True:
                labels, angle = labels

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            loss_canonical.append(loss.detach().numpy())

    return loss_canonical

# Function to evaluate the model
def evaluate_model(model, test_loader, tasks_flattened=False):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            if tasks_flattened is True:
                labels, angle = labels

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return accuracy