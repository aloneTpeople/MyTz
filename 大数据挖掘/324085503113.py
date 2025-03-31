import math


def factorial(n):
    """计算非负整数的阶乘"""
    if not isinstance(n, int):
        raise TypeError("必须是整数")
    if n < 0:
        raise ValueError("必须是非负整数")
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result


def is_prime(n):
    """判断整数是否为素数"""
    if not isinstance(n, int):
        raise TypeError("必须是整数")
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    max_divisor = int(math.sqrt(n)) + 1
    for i in range(3, max_divisor, 2):
        if n % i == 0:
            return False
    return True


def fibonacci(n):
    """生成斐波那契数列"""
    if not isinstance(n, int) or n < 0:
        raise ValueError("必须是非负整数")
    sequence = []
    a, b = 0, 1
    for _ in range(n):
        sequence.append(a)
        a, b = b, a + b
    return sequence


def main():
    """主交互程序"""
    while True:
        print("\n" + "=" * 30)
        print("功能菜单:")
        print("1. 计算阶乘")
        print("2. 判断素数")
        print("3. 生成斐波那契数列")
        print("4. 退出程序")
        print("=" * 30)

        choice = input("请输入选项 (1-4): ").strip()

        if choice == "4":
            print("程序已退出！")
            break

        if choice not in ("1", "2", "3"):
            print("无效选项，请重新输入！")
            continue

        try:
            if choice == "1":
                num = int(input("请输入非负整数: "))
                print(f"{num}! = {factorial(num)}")

            elif choice == "2":
                num = int(input("请输入整数: "))
                if is_prime(num):
                    print(f"{num} 是素数")
                else:
                    print(f"{num} 不是素数")

            elif choice == "3":
                length = int(input("请输入数列长度: "))
                fib = fibonacci(length)
                print(f"斐波那契数列（前{length}项）: {fib}")

        except ValueError as e:
            print(f"输入错误: {e}")
        except TypeError as e:
            print(f"类型错误: {e}")
        except Exception as e:
            print(f"发生意外错误: {e}")


if __name__ == "__main__":
    main()