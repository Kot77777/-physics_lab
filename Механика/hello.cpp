#include <iostream>
using namespace std;
int main(){
	int* array = new int[2];
	array[2] = 47;
	cout << "hello world";
	delete[] array;
	return 0;
}
