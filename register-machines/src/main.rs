use std::collections::HashMap;

type Label = usize;
type Register = u128;
type State = (Label, HashMap<Register, u128>);

#[derive(Clone, Copy, Debug, PartialEq)]
enum Instruction {
    Add(Register, Label),
    Sub(Register, Label, Label),
    Halt
}

use Instruction::*;

fn eval_program(program: &[Instruction], init: &State) -> State {
    let mut state = init.clone();
    let (label, registers) = &mut state;

    loop {
        /* Reveal effects of instruction to state. */
        match program[*label] {
            Add(reg, l) => {
                /* Increment target register. */
                *registers.get_mut(&reg).unwrap() += 1;
    
                /* Jump to label. */
                *label = l;
            },
            Sub(reg, lsub, lnop) => {
                let register = registers.get_mut(&reg).unwrap();
    
                if *register == 0 {
                    /* Register can't be decremented. */
                    *label = lnop;
                } else {
                    /* Decrement register. */
                    *register -= 1;
                    *label = lsub;
                }
            },
            Halt => { break; },
        };
    }

    return state;
}
// <<x,y>> = (2^x)*(2y+1)
// NULLABLE
fn encode_pair1(x: u128, y: u128) -> u128 {
    return (1 << x) * (2 * y + 1);
}
// <x,y> = (2^x)*(2y+1)-1
// NOT NULLABLE // NORMAL ONE
fn encode_pair2(x: u128, y: u128) -> u128 {
    return encode_pair1(x, y) - 1;
}
fn encode_list_to_godel(l: &[u128]) -> u128 {
    return l.iter().rev()
        .fold(0, |x, &y| encode_pair1(y, x));
}
fn encode_instruction(instruction: &Instruction) -> u128 {
    return match *instruction {
        Add(i, j) => encode_pair1(2 * i, j as u128),
        Sub(i, j, k) => encode_pair1(2 * i + 1, encode_pair2(j as u128, k as u128)),
        Halt => 0
    }
}

fn encode_program_to_list(program: &[Instruction]) -> Vec<u128> {
    return program.iter().map(encode_instruction).collect();
}
fn count_trailing_zeros(mut n: u128) -> u128 {
    // PRE: n /= 0
    let mut zeros = 0;

    while n % 2 == 0 {
        zeros += 1;
        n >>= 1;
    }

    return zeros;
}
// a = (2^x)*(2y+1)
// NULLABLE
fn decode_pair1(a: u128) -> (u128, u128) {
    return decode_pair2(a - 1);
}
// a = (2^x)*(2y+1)-1
// NOT NULLABLE
fn decode_pair2(a: u128) -> (u128, u128) {
    let x = count_trailing_zeros(a + 1);
    let y = (a + 1) >> (x + 1);

    return (x, y);
}
fn decode_godel_to_list(mut g: u128) -> Vec<u128> {
    let mut list = vec!();

    while g != 0 {
        let (head, tail) = decode_pair1(g);

        g = tail;

        list.push(head);
    }

    return list;
}
fn decode_instruction(n: u128) -> Instruction {
    if n == 0 { return Halt; }

    let (f, arg) = decode_pair1(n);

    if f % 2 == 0 {
        // EVEN: add
        return Add(f / 2, arg as usize);
    } else {
        // ODD: sub
        let (j, k) = decode_pair2(arg);
        return Sub((f - 1) / 2, j as usize, k as usize);
    }
}
fn decode_list_to_program(program: &[u128]) -> Vec<Instruction> {
    return program.iter().map(|&n| decode_instruction(n)).collect()
}

fn main() {
    println!("{}", encode_instruction(&Sub(0, 2, 1)));
}

mod test {
    use crate::*;
    #[test]
    fn godel_num_to_godel_list() {
        let n = 2u128.pow(46) * 20483;
        let godel_list = decode_godel_to_list(n);
        let true_godel_list = vec![46, 0, 10, 1];
        assert_eq!(godel_list, true_godel_list)
    }

    #[test]
    fn godel_list_to_godel_num() {
        let godel_num = encode_list_to_godel(&[46, 0, 10, 1]);
        assert_eq!(godel_num, 2u128.pow(46) * 20483)
    }

    #[test]
    fn godel_list_to_program() {
        let program = decode_list_to_program(&vec![46, 0, 10, 1]);
        assert_eq!(program, vec![Sub(0, 2, 1), Halt, Sub(0, 0, 1), Add(0, 0)])
    }

    #[test]
    fn program_to_godel_list() {
        let program = encode_program_to_list(&[Sub(0, 2, 1), Halt, Sub(0, 0, 1), Add(0, 0)]);
        assert_eq!(program, [46, 0, 10, 1])
    }

    #[test]
    fn program_produces_correct_state() {
        use std::array::IntoIter;
        let program = vec![
            Sub(1, 2, 1),
            Halt,
            Sub(1, 3, 4),
            Sub(1, 5, 4),
            Halt,
            Add(0, 0)
        ];
        let final_state = eval_program(
            &program,
            &(
                0,
                HashMap::<_, _>::from_iter(IntoIter::new([(0, 0), (1, 7)]))
            ),
        );
        assert_eq!(
            final_state,
            (
                4,
                HashMap::<_, _>::from_iter(IntoIter::new([(0, 2), (1, 0)]))
            )
        )
    }
}