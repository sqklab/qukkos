
grammar xasm_single;

line
   : statement 
   | comment
   ;

/* A program statement */
statement
   : qinst 
   | cinst
   ;

/* A program comment */
comment
   : COMMENT
   ;

qinst
   : inst_name=id '(' explist ')' ';'
   ;

cinst
   : 'const'? type_name=cpp_type var_name=exp ('=' var_value=exp)? ';'
   | exp '++' ';'
   | exp '--' ';'
   | 'for' '(' cpp_type exp '=' exp ';' (exp compare exp)? ';' ((exp ('++' | '--')) | (('++' | '--') exp))?  ')' '{'?
   | '}'
   | exp '(' explist? ')' ';'
   | 'if' '(' explist ')' '{'?
   | 'else' '{'?
   | 'const'? type_name=cpp_type var_name=exp '=' '(' exp '==' exp ')' '?' exp ':' exp ';'
   | 'break' ';'
   | 'continue' ';'
   | 'return' ';'
   | exp '=' exp ';'
   ;

cpp_type 
   : 'auto' ('&'|'*')?
   | exp
   ;

compare 
   : '>' | '<' | '>=' | '<=' ;

explist
   : exp ( ',' exp )*
   ;

exp
   : id
   | exp '+' exp
   | exp '-' exp
   | exp '*' exp
   | exp '/' exp
   | exp '::' exp
   | exp '<<' exp
   | exp '<' exp '>'
   | exp '::' exp '(' explist ')'
   | exp '.' exp '(' ')'
   | exp '.' exp '(' explist ')'
   | '-'exp
   | exp '^' exp
   | '(' exp ')'
   | '{' explist '}'
   | unaryop '(' exp ')'
   | exp '(' explist? ')'
   | exp '[' exp ']'
   | string
   | real
   | INT
   | CHAR
   | 'pi'
   | exp '&&' exp
   | exp '||' exp
   | '!' exp
   ;

unaryop
   : 'sin'
   | 'cos'
   | 'tan'
   | 'exp'
   | 'ln'
   | 'sqrt'
   ;

id
   : ID
   ;

real
   : REAL
   ;

string
   : STRING
   ;

COMMENT
   : '//' ~ [\r\n]* EOL
   ;

ID
   : [a-z][A-Za-z0-9_]*
   | [A-Z][A-Za-z0-9_]*
   | [A-Z][A-Za-z]*
   ;

REAL
   : INT? ( '.' (INT) )
   ;

INT
   : ('0'..'9')+
   ;

CHAR
   : '\'' ~ ['] '\''
   | '\'\\\'\''
   ;

STRING
   : '"' ~ ["]* '"'
   ;

WS
   : [ \t\r\n] -> skip
   ;

EOL
: '\r'? '\n'
;
