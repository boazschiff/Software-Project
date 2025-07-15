cat << 'EOF' > install_valgrind.sh
#!/bin/bash

set -e  # Exit on error

echo "🔧 Installing dependencies..."
sudo apt update
sudo apt install -y build-essential wget

echo "⬇️ Downloading Valgrind 3.25.1..."
wget https://sourceware.org/pub/valgrind/valgrind-3.25.1.tar.bz2

echo "📦 Extracting..."
tar -xf valgrind-3.25.1.tar.bz2
cd valgrind-3.25.1

echo "⚙️ Configuring..."
./configure

echo "🛠️ Building..."
make -j$(nproc)

echo "🚀 Installing..."
sudo make install

echo "✅ Done! Installed version:"
valgrind --version
EOF
