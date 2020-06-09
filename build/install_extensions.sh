for extension in `cat ./build/extensions.txt`
    do code-server --install-extension ${extension}
done