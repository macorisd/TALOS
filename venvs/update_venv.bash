if [[ -n "$VIRTUAL_ENV" ]]; then
    deactivate
fi

rm -rf ./talos_env/

python3 -m venv talos_env

source ./talos_env/bin/activate

echo "✅ Virtual environment activated:"
echo "   VIRTUAL_ENV = $VIRTUAL_ENV"
echo "   Python path = $(which python)"
echo

read -p "Press Enter to install requirements from ../requirements.txt (or Ctrl+C to cancel)..."

pip install -r ../requirements.txt

echo "✅ Requirements installed."
