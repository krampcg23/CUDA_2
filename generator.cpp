#include <iostream>
#include <fstream>
#include <vector>
using namespace std;

int main() {
    ofstream file;
    file.open("sm3.txt");
    vector<vector<int> > v;
    int c = 20001;
    int r = 4003;
    int sum = 0;
    for (int i = 0; i < r; i++) {
        for (int j = 0; j < c; j++) {
            if ((rand() % 100) == 1) {
                vector<int> vec = {i, j, 1};
                v.push_back(vec);
                sum++;
            }
        }
    }
    file << r << " " << c << " " << sum << endl;
    for (auto i : v) {
        file << i[0]+1 << " " << i[1]+1 << " " << i[2] << endl;
    }

    file.close();

    return 0;
}
